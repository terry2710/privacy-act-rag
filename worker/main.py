import json
import logging

import boto3

from api.config import get_settings

logger = logging.getLogger("worker")


def handle_message(body):
    logger.info("received ingestion message for document %s", body.get("document_id"))


def run_forever():
    settings = get_settings()
    sqs = boto3.client("sqs", endpoint_url=settings.sqs_endpoint_url)
    queue_url = sqs.get_queue_url(QueueName=settings.sqs_queue_name)["QueueUrl"]
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20
        )
        for message in response.get("Messages", []):
            handle_message(json.loads(message["Body"]))
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
