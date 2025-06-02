from azure.storage.blob import BlobServiceClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import os
import json
import azure.functions as func

# Azure Blob Storage configuration
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# Azure Custom Vision configuration
CUSTOM_VISION_ENDPOINT = os.getenv("CUSTOM_VISION_ENDPOINT")
CUSTOM_VISION_PROJECT_ID = os.getenv("CUSTOM_VISION_PROJECT_ID")
CUSTOM_VISION_PREDICTION_KEY = os.getenv("CUSTOM_VISION_PREDICTION_KEY")

# Azure Service Bus configuration
SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
SERVICE_BUS_TOPIC_NAME = os.getenv("SERVICE_BUS_TOPIC_NAME")

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Get the uploaded file
        file = req.files['file']
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=file.filename)

        # Upload the image to Blob Storage
        blob_client.upload_blob(file.stream, overwrite=True)

        # Classify the image using Azure Custom Vision
        prediction_client = CustomVisionPredictionClient(CUSTOM_VISION_ENDPOINT, CUSTOM_VISION_PREDICTION_KEY)
        with open(file.stream, "rb") as image_data:
            results = prediction_client.classify_image(CUSTOM_VISION_PROJECT_ID, "Iteration1", image_data)

        # Prepare the classification result
        classification_result = {tag.name: tag.probability for tag in results.predictions}
        
        # Send the result to Service Bus
        service_bus_client = ServiceBusClient.from_connection_string(SERVICE_BUS_CONNECTION_STRING)
        with service_bus_client:
            sender = service_bus_client.get_topic_sender(SERVICE_BUS_TOPIC_NAME)
            with sender:
                message = ServiceBusMessage(json.dumps(classification_result))
                sender.send_messages(message)

        return func.HttpResponse("Image uploaded and classified successfully.", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)