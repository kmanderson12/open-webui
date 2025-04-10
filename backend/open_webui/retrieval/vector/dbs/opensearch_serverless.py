from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
from typing import Optional
import logging
import time

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import (
    OPENSEARCH_SERVERLESS_REGION,
    OPENSEARCH_SERVERLESS_HOST,
    OPENSEARCH_SERVERLESS_PORT,
    OPENSEARCH_SERVERLESS_SERVICE_NAME,
    OPENSEARCH_SERVERLESS_INDEX_CREATION_DELAY
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

        # Test the connection
        try:
            indices = self.client.indices.get("*")  # Get all indices
            logger.info(f"Successfully connected to OpenSearch Serverless. Found {len(indices)} indices.")
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch Serverless: {str(e)}")

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

    def _get_opensearch_id_for_original_id(self, collection_name: str, original_id: str, timeout: int = 30) -> Optional[str]:
        """Get OpenSearch ID for an original ID."""
        query = {
            "query": {"term": {"original_id": original_id}},
            "_source": False
        }

        try:
            result = self.client.search(
                index=self._get_index_name(collection_name),
                body=query,
                size=1,
                request_timeout=timeout
            )

            if result["hits"]["hits"]:
                return result["hits"]["hits"][0]["_id"]
            else:
                logger.warning(f"No document found with original_id: {original_id}")
                return None

        except Exception as e:
            logger.error(f"Error getting OpenSearch ID for {original_id}: {str(e)}")
            return None

    def has_collection(self, collection_name: str) -> bool:
        try:
            return self.client.indices.exists(index=self._get_index_name(collection_name))
        except Exception as e:
            logger.error(f"Error checking if collection {collection_name} exists: {str(e)}")
            return False

    def delete_collection(self, collection_name: str):
        try:
            self.client.indices.delete(index=self._get_index_name(collection_name))
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

        size = limit if limit else 10000 # Increase size to account for large files

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

    def _create_index_if_not_exists(self, collection_name: str, dimension: int, delay: float = 0):
        if not self.has_collection(collection_name):
            self._create_index(collection_name, dimension)
            time.sleep(delay) # Optional delay to avoid race conditions

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
            # Ensure the index exists
            if items:
                self._create_index_if_not_exists(
                    collection_name=collection_name, dimension=len(items[0]["vector"]), delay=OPENSEARCH_SERVERLESS_INDEX_CREATION_DELAY
                )

                # Process items individually for clarity
                for item in items:
                    self.client.index(
                        index=self._get_index_name(collection_name),
                        body={
                            "original_id": item["id"],
                            "vector": item["vector"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                        }
                    )
                    logger.info(f"Inserted item with ID {item['id']} into {collection_name}")
            else:
                logger.warning("No items to insert")

        except Exception as e:
            logger.error(f"Error inserting into collection {collection_name}: {str(e)}")
            raise

    def upsert(self, collection_name: str, items: list[VectorItem]):
        try:
            # Process each item individually for clarity
            for item in items:
                # Check if the item exists
                opensearch_id = self._get_opensearch_id_for_original_id(collection_name, item["id"])

                if opensearch_id:
                    # Item exists - update it
                    logger.info(f"Updating existing item with ID {item['id']} (OpenSearch ID: {opensearch_id})")
                    self.client.update(
                        index=self._get_index_name(collection_name),
                        id=opensearch_id,
                        body={
                            "doc": {
                                "vector": item["vector"],
                                "text": item["text"],
                                "metadata": item["metadata"],
                            }
                        }
                    )
                else:
                    # Item doesn't exist - create the index if needed
                    if not self.has_collection(collection_name):
                        self._create_index(collection_name, dimension=len(item["vector"]))

                    # Insert new item
                    logger.info(f"Inserting new item with ID {item['id']}")
                    self.client.index(
                        index=self._get_index_name(collection_name),
                        body={
                            "original_id": item["id"],
                            "vector": item["vector"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                        }
                    )

        except Exception as e:
            logger.error(f"Error upserting in collection {collection_name}: {str(e)}")
            raise

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        max_documents: int = 10000  # Safety limit for total documents to delete
        batch_size: int = 100       # Number of documents per batch
        request_timeout: int = 30   # Timeout for OpenSearch requests in seconds
        batch_delay: float = 1.0    # Delay between batches in seconds

        try:
            # Check if collection exists
            if not self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist. Nothing to delete.")
                return

            if ids:
                # Process each ID individually
                deleted_count = 0
                not_found_count = 0

                for original_id in ids:
                    # Find the OpenSearch ID
                    opensearch_id = self._get_opensearch_id_for_original_id(
                        collection_name,
                        original_id,
                        timeout=request_timeout
                    )

                    if opensearch_id:
                        try:
                            # Delete the document
                            logger.info(f"Deleting item with ID {original_id} (OpenSearch ID: {opensearch_id})")
                            self.client.delete(
                                index=self._get_index_name(collection_name),
                                id=opensearch_id,
                                request_timeout=request_timeout
                            )
                            deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error deleting document {original_id}: {str(e)}")
                    else:
                        not_found_count += 1
                        logger.warning(f"Could not find OpenSearch ID for original_id {original_id} - nothing deleted")

                logger.info(f"ID-based deletion complete: {deleted_count} deleted, {not_found_count} not found")

            elif filter:
                # Validate filter to prevent accidentally deleting all documents
                if not filter:
                    logger.warning("Empty filter provided. This would match all documents. Operation aborted.")
                    return

                # Since delete_by_query and scroll are not available, we'll use pagination
                total_deleted = 0
                max_iterations = (max_documents // batch_size) + 1  # Calculate max iterations based on document limit
                iteration = 0

                while iteration < max_iterations and total_deleted < max_documents:
                    iteration += 1

                    # Construct the query to find documents matching the filter
                    query_body = {
                        "query": {"bool": {"filter": []}},
                        "_source": False,  # We only need the IDs
                        "size": min(batch_size, max_documents - total_deleted)  # Respect max_documents limit
                    }

                    for field, value in filter.items():
                        query_body["query"]["bool"]["filter"].append(
                            {"match": {"metadata." + str(field): value}}
                        )

                    # Execute search to get documents to delete
                    logger.info(f"Searching for documents to delete (batch {iteration})")
                    logger.debug(f"Query: {query_body}")

                    try:
                        result = self.client.search(
                            index=self._get_index_name(collection_name),
                            body=query_body,
                            request_timeout=request_timeout
                        )
                    except Exception as e:
                        logger.error(f"Error searching for documents to delete: {str(e)}")
                        break

                    hits = result["hits"]["hits"]
                    if not hits:
                        logger.info(f"No more documents found matching the filter. Total deleted: {total_deleted}")
                        break

                    # Create bulk delete actions
                    bulk_actions = []
                    for hit in hits:
                        bulk_actions.append({
                            "delete": {
                                "_index": self._get_index_name(collection_name),
                                "_id": hit["_id"]
                            }
                        })

                    # Execute bulk delete without refresh parameter
                    if bulk_actions:
                        try:
                            bulk_result = self.client.bulk(
                                body=bulk_actions,
                                request_timeout=request_timeout
                                # No refresh=True parameter since it's not supported
                            )

                            # Detailed error checking for bulk operations
                            if bulk_result.get("errors", False):
                                error_items = [
                                    item for item in bulk_result.get("items", [])
                                    if item.get("delete", {}).get("status", 200) >= 400
                                ]
                                error_count = len(error_items)
                                if error_count > 0:
                                    logger.warning(
                                        f"{error_count} bulk delete operations failed. "
                                        f"First few errors: {error_items[:3]}"
                                    )

                            batch_deleted = len(bulk_actions) - (len(error_items) if 'error_items' in locals() else 0)
                            total_deleted += batch_deleted
                            logger.info(f"Deleted batch of {batch_deleted} documents, total deleted: {total_deleted}")

                        except Exception as e:
                            logger.error(f"Error executing bulk delete: {str(e)}")
                            break

                    # Add a delay between batches since we can't refresh the index
                    # This gives OpenSearch Serverless time to process the deletions
                    time.sleep(batch_delay)

                    # If we got fewer documents than the batch size, we're done
                    if len(hits) < batch_size:
                        logger.info(f"Completed deletion. Total deleted: {total_deleted}")
                        break

                    # Safety check for reaching document limit
                    if total_deleted >= max_documents:
                        logger.warning(
                            f"Reached maximum document limit ({max_documents}). "
                            f"Stopping deletion to prevent excessive operations."
                        )
                        break

                if iteration >= max_iterations:
                    logger.warning(
                        f"Reached maximum iteration limit ({max_iterations}). "
                        f"Stopping deletion after {total_deleted} documents."
                    )

        except Exception as e:
            logger.error(f"Error deleting from collection {collection_name}: {str(e)}")
            raise

    def reset(self):
        try:
            indices = self.client.indices.get(index=f"{self.index_prefix}_*")
            for index in indices:
                logger.info(f"Deleting index {index}")
                self.client.indices.delete(index=index)
            logger.info("Reset all collections")
        except Exception as e:
            logger.error(f"Error resetting collections: {str(e)}")
            raise