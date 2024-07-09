import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def upload_folder_to_blob(connection_string, container_name, local_folder_path):
    """
    Function to upload all files in a local folder to Azure Blob Storage

    :param connection_string: Azure Storage connection string
    :param container_name: Name of the destination Blob container
    :param local_folder_path: Path of the local folder to upload
    """
    # Create Blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Create container client
    container_client = blob_service_client.get_container_client(container_name)

    # Upload all files in the local folder
    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            # Full path of the file
            file_path = os.path.join(root, file)

            # Path of the Blob in the container
            blob_path = os.path.relpath(file_path, local_folder_path)

            # Create Blob client
            blob_client = container_client.get_blob_client(blob_path)

            # Upload the file
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded {file_path} to {blob_path}")
