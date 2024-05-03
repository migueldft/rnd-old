import boto3

client = boto3.client('sagemaker-runtime')

custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
endpoint_name = "second-order"                              # Your endpoint name.
content_type = "application/json"                           # The MIME type of the input data in the request body.
accept = "application/json"                                 # The desired MIME type of the inference in the response.
with open('ml/input/api/payload.json', 'r') as file:
    payload = file.read()                                   # Payload for inference.

response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    CustomAttributes=custom_attributes,
    ContentType=content_type,
    Accept=accept,
    Body=payload
)

response_body = response['Body']                # If model receives and updates the custom_attributes header
print(response_body.read().decode("utf-8"))     # by adding "Trace id: " in front of custom_attributes in the request,
                                                # custom_attributes in response becomes
                                                # "Trace ID: c000b4f9-df62-4c85-a0bf-7c525f9104a4"
