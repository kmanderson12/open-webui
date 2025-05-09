#!/bin/bash
# push-to-ecr.sh

# Set variables
AWS_REGION="us-gov-west-1"
AWS_ACCOUNT_ID="084362420086"
LOCAL_IMAGE="open-webui-custom"
LOCAL_IMAGE_TAG="latest"
REPOSITORY_NAME="open-webui"
IMAGE_TAG="aoss-client"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag the image
docker tag $LOCAL_IMAGE:$LOCAL_IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

# Push the image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

echo "Successfully pushed $REPOSITORY_NAME:$IMAGE_TAG to ECR"
