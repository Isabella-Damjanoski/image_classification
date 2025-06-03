import logging
import os
import json
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import azure.functions as func
from dotenv import load_dotenv
load_dotenv()

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION) 

@app.function_name(name="catdogclassifier") 
@app.blob_trigger(
    arg_name="myblob",
    path="imageblob/{name}",
    connection="AzureWebJobsStorage") 
def catdogclassifier(myblob: func.InputStream): 
    logging.info(f"Blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n" 
                 f"Blob Size: {myblob.length} bytes") 

    logging.info("Function triggered by blob upload.")

    # Read image data from blob (works for jpg, png, etc.)
    image_data = myblob.read()

    # Custom Vision configuration from environment
    endpoint = os.getenv("CUSTOM_VISION_ENDPOINT")
    project_id = os.getenv("CUSTOM_VISION_PROJECT_ID")  # Should be GUID only!
    prediction_key = os.getenv("CUSTOM_VISION_PREDICTION_KEY")
    iteration_name = os.getenv("CUSTOM_VISION_ITERATION_NAME")  # Optional: set in settings

    # Authenticate and predict
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    prediction_client = CustomVisionPredictionClient(endpoint, credentials)
    results = prediction_client.classify_image(
        project_id, iteration_name, image_data
    )

    # Find the top prediction (cat or dog)
    top_prediction = max(results.predictions, key=lambda x: x.probability)
    result = {
        "blob_name": myblob.name,
        "prediction": top_prediction.tag_name,
        "probability": top_prediction.probability
    }

    # Send result to Service Bus
    servicebus_conn_str = os.getenv("SERVICE_BUS_CONNECTION_STRING")
    topic_name = os.getenv("SERVICE_BUS_TOPIC_NAME")
    with ServiceBusClient.from_connection_string(servicebus_conn_str) as client:
        sender = client.get_topic_sender(topic_name)
        with sender:
            message = ServiceBusMessage(json.dumps(result))
            sender.send_messages(message)
            
    logging.info("Classification result sent to Service Bus.")