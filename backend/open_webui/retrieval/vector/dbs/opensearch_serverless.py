from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
from opensearchpy.helpers import bulk
from typing import Optional
import logging
import time

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import (
    OPENSEARCH_SERVERLESS_REGION,
    OPENSEARCH_SERVERLESS_HOST,
    OPENSEARCH_SERVERLESS_PORT,
    OPENSEARCH_SERVERLESS_SERVICE_NAME,
)

# Set up logging
logger = logging.getLogger(__name__)


class OpenSearchServerlessClient:
    def __init__(self):
        self.index_prefix = "open_webui"
        # Create AWS credentials and signer
        credentials = boto3.Session().get_credentials()
        region = OPENSEARCH_SERVERLESS_REGION

        # Create the auth for serverless using IAM role
        auth = AWSV4SignerAuth(credentials, region, OPENSEARCH_SERVERLESS_SERVICE_NAME)

        # Clean up the host URL - remove protocol if present
        host = OPENSEARCH_SERVERLESS_HOST
        if host.startswith("https://"):
            host = host[8:]
        elif host.startswith("http://"):
            host = host[7:]

        logger.info(f"Connecting to OpenSearch Serverless host: {host}")

        # Initialize the client with AWS auth
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': OPENSEARCH_SERVERLESS_PORT}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

        # Use a real in-memory cache for ID mappings
        self.id_mapping_cache = {}  # collection_name -> {original_id -> opensearch_id}

        # Test the connection
        try:
            info = self.client.info()
            logger.info(f"Successfully connected to OpenSearch Serverless. Cluster: {info.get('cluster_name', 'unknown')}, Version: {info.get('version', {}).get('number', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch Serverless: {str(e)}")
            # You can choose to raise the exception or continue
            # raise  # Uncomment if you want to fail fast when connection fails

    def _get_index_name(self, collection_name: str) -> str:
        return f"{self.index_prefix}_{collection_name}"

    def _result_to_get_result(self, result) -> GetResult:
        if not result["hits"]["hits"]:
            return None

        ids = []
        documents = []
        metadatas = []

        for hit in result["hits"]["hits"]:
            # Use our mapped ID rather than OpenSearch ID
            original_id = hit["_source"].get("original_id")
            ids.append(original_id)
            documents.append(hit["_source"].get("text"))
            metadatas.append(hit["_source"].get("metadata"))

        return GetResult(ids=[ids], documents=[documents], metadatas=[metadatas])

    def _result_to_search_result(self, result) -> SearchResult:
        if not result["hits"]["hits"]:
            return None

        ids = []
        distances = []
        documents = []
        metadatas = []

        for hit in result["hits"]["hits"]:
            # Use our mapped ID rather than OpenSearch ID
            original_id = hit["_source"].get("original_id")
            ids.append(original_id)
            distances.append(hit["_score"])
            documents.append(hit["_source"].get("text"))
            metadatas.append(hit["_source"].get("metadata"))

        return SearchResult(
            ids=[ids],
            distances=[distances],
            documents=[documents],
            metadatas=[metadatas],
        )

    def _create_index(self, collection_name: str, dimension: int):
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "original_id": {"type": "keyword"},  # Store our ID
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "index": True,
                        "similarity": "faiss",
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16,
                            },
                        },
                    },
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                }
            },
        }
        try:
            self.client.indices.create(
                index=self._get_index_name(collection_name), body=body
            )
            logger.info(f"Created index {self._get_index_name(collection_name)}")
        except Exception as e:
            logger.error(f"Error creating index {self._get_index_name(collection_name)}: {str(e)}")
            raise

    def _create_batches(self, items: list[VectorItem], batch_size=100):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _get_opensearch_id(self, collection_name: str, original_id: str) -> Optional[str]:
        """Get OpenSearch ID for an original ID with caching."""
        # Check cache first
        if (collection_name in self.id_mapping_cache and
                original_id in self.id_mapping_cache[collection_name]):
            return self.id_mapping_cache[collection_name][original_id]

        # Cache miss - query OpenSearch
        query = {
            "query": {"term": {"original_id": original_id}},
            "_source": False  # We only need the ID, not the source
        }

        try:
            result = self.client.search(
                index=self._get_index_name(collection_name),
                body=query,
                size=1
            )

            if result["hits"]["hits"]:
                opensearch_id = result["hits"]["hits"][0]["_id"]
                # Update cache
                if collection_name not in self.id_mapping_cache:
                    self.id_mapping_cache[collection_name] = {}
                self.id_mapping_cache[collection_name][original_id] = opensearch_id
                return opensearch_id
        except Exception as e:
            logger.error(f"Error getting OpenSearch ID for {original_id}: {str(e)}")

        return None

    def _batch_get_opensearch_ids(self, collection_name: str, original_ids: list[str]) -> dict:
        """Get OpenSearch IDs for multiple original IDs in one query."""
        if not original_ids:
            return {}

        # Check cache first
        result = {}
        missing_ids = []

        if collection_name in self.id_mapping_cache:
            cache = self.id_mapping_cache[collection_name]
            for orig_id in original_ids:
                if orig_id in cache:
                    result[orig_id] = cache[orig_id]
                else:
                    missing_ids.append(orig_id)
        else:
            self.id_mapping_cache[collection_name] = {}
            missing_ids = original_ids

        # If all IDs were in cache, return immediately
        if not missing_ids:
            return result

        # Query for missing IDs
        query = {
            "query": {
                "terms": {
                    "original_id": missing_ids
                }
            },
            "_source": ["original_id"],
            "size": len(missing_ids)
        }

        try:
            search_result = self.client.search(
                index=self._get_index_name(collection_name),
                body=query
            )

            # Update cache and result
            for hit in search_result["hits"]["hits"]:
                orig_id = hit["_source"]["original_id"]
                os_id = hit["_id"]
                self.id_mapping_cache[collection_name][orig_id] = os_id
                result[orig_id] = os_id
        except Exception as e:
            logger.error(f"Error batch getting OpenSearch IDs: {str(e)}")

        return result

    def _invalidate_id_cache(self, collection_name: str, original_id: str = None):
        """Invalidate the ID mapping cache for a collection or specific ID."""
        if original_id and collection_name in self.id_mapping_cache:
            if original_id in self.id_mapping_cache[collection_name]:
                del self.id_mapping_cache[collection_name][original_id]
        elif collection_name in self.id_mapping_cache:
            del self.id_mapping_cache[collection_name]

    def has_collection(self, collection_name: str) -> bool:
        try:
            return self.client.indices.exists(index=self._get_index_name(collection_name))
        except Exception as e:
            logger.error(f"Error checking if collection {collection_name} exists: {str(e)}")
            return False

    def delete_collection(self, collection_name: str):
        try:
            self.client.indices.delete(index=self._get_index_name(collection_name))
            # Clear cache for this collection
            self._invalidate_id_cache(collection_name)
            logger.info(f"Deleted index {self._get_index_name(collection_name)}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            raise

    def search(
        self, collection_name: str, vectors: list[list[float | int]], limit: int
    ) -> Optional[SearchResult]:
        try:
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return None

            # Use script_score with knn_score for cosine similarity
            query = {
                "size": limit,
                "_source": ["text", "metadata", "original_id"],
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {
                                "field": "vector",
                                "query_value": vectors[0],
                                "space_type": "cosinesimil"
                            }
                        }
                    }
                }
            }

            result = self.client.search(
                index=self._get_index_name(collection_name), body=query
            )

            return self._result_to_search_result(result)

        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {str(e)}")
            return None

    def query(
        self, collection_name: str, filter: dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        if not self.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return None

        query_body = {
            "query": {"bool": {"filter": []}},
            "_source": ["text", "metadata", "original_id"],
        }

        for field, value in filter.items():
            query_body["query"]["bool"]["filter"].append(
                {"match": {"metadata." + str(field): value}}
            )

        size = limit if limit else 10

        try:
            result = self.client.search(
                index=self._get_index_name(collection_name),
                body=query_body,
                size=size,
            )

            return self._result_to_get_result(result)

        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {str(e)}")
            return None

    def _create_index_if_not_exists(self, collection_name: str, dimension: int):
        if not self.has_collection(collection_name):
            self._create_index(collection_name, dimension)

    def get(self, collection_name: str) -> Optional[GetResult]:
        try:
            query = {"query": {"match_all": {}}, "_source": ["text", "metadata", "original_id"]}

            result = self.client.search(
                index=self._get_index_name(collection_name), body=query
            )
            return self._result_to_get_result(result)
        except Exception as e:
            logger.error(f"Error getting all documents from collection {collection_name}: {str(e)}")
            return None

    def insert(self, collection_name: str, items: list[VectorItem]):
        try:
            self._create_index_if_not_exists(
                collection_name=collection_name, dimension=len(items[0]["vector"])
            )

            for batch in self._create_batches(items):
                actions = [
                    {
                        "_op_type": "index",
                        "_index": self._get_index_name(collection_name),
                        # No _id field - let OpenSearch generate it
                        "_source": {
                            "original_id": item["id"],  # Store our ID in the document
                            "vector": item["vector"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                        },
                    }
                    for item in batch
                ]
                bulk(self.client, actions)
                logger.info(f"Inserted batch of {len(batch)} items into {collection_name}")
        except Exception as e:
            logger.error(f"Error inserting into collection {collection_name}: {str(e)}")
            raise

    def upsert(self, collection_name: str, items: list[VectorItem]):
        try:
            self._create_index_if_not_exists(
                collection_name=collection_name, dimension=len(items[0]["vector"])
            )

            # Split items into two categories: new items and updates
            # First, query all original_ids in bulk
            original_ids = [item["id"] for item in items]
            existing_ids_map = self._batch_get_opensearch_ids(collection_name, original_ids)

            # Prepare bulk actions for both new and existing items
            actions = []

            for item in items:
                opensearch_id = existing_ids_map.get(item["id"])
                if opensearch_id:
                    # Update
                    actions.append({
                        "_op_type": "update",
                        "_index": self._get_index_name(collection_name),
                        "_id": opensearch_id,
                        "doc": {
                            "vector": item["vector"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                        }
                    })
                else:
                    # Insert
                    actions.append({
                        "_op_type": "index",
                        "_index": self._get_index_name(collection_name),
                        "_source": {
                            "original_id": item["id"],
                            "vector": item["vector"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                        }
                    })

            # Execute bulk operation
            if actions:
                bulk(self.client, actions)
                logger.info(f"Upserted {len(items)} items in {collection_name}")
        except Exception as e:
            logger.error(f"Error upserting in collection {collection_name}: {str(e)}")
            raise

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        try:
            if ids:
                # Get all OpenSearch IDs in one query
                opensearch_ids = []
                id_mapping = self._batch_get_opensearch_ids(collection_name, ids)

                for original_id, opensearch_id in id_mapping.items():
                    opensearch_ids.append(opensearch_id)
                    # Invalidate cache
                    self._invalidate_id_cache(collection_name, original_id)

                if opensearch_ids:
                    actions = [
                        {
                            "_op_type": "delete",
                            "_index": self._get_index_name(collection_name),
                            "_id": os_id,
                        }
                        for os_id in opensearch_ids
                    ]
                    bulk(self.client, actions)
                    logger.info(f"Deleted {len(opensearch_ids)} items from {collection_name}")
            elif filter:
                query_body = {
                    "query": {"bool": {"filter": []}},
                }
                for field, value in filter.items():
                    query_body["query"]["bool"]["filter"].append(
                        {"match": {"metadata." + str(field): value}}
                    )
                self.client.delete_by_query(
                    index=self._get_index_name(collection_name), body=query_body
                )
                # After delete, invalidate the entire collection cache
                self._invalidate_id_cache(collection_name)
                logger.info(f"Deleted items by filter from {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting from collection {collection_name}: {str(e)}")
            raise

    def reset(self):
        try:
            indices = self.client.indices.get(index=f"{self.index_prefix}_*")
            for index in indices:
                self.client.indices.delete(index=index)
            # Clear all caches
            self.id_mapping_cache = {}
            logger.info("Reset all collections")
        except Exception as e:
            logger.error(f"Error resetting collections: {str(e)}")
            raise