import os
import requests
import yaml


def get_local_version(version_file):
    """read version from a local yaml file and return"""
    with open(version_file, 'r') as file:
        data = yaml.safe_load(file)
        return data.get('version')


def update_local_version(new_version, version_file):
    """update the version number to the local yaml file"""
    with open(version_file, 'r') as file:
        data = yaml.safe_load(file)
    data['version'] = new_version
    with open(version_file, 'w') as file:
        yaml.safe_dump(data, file)


def get_latest_version_from_github(model_url):
    """Fetches the latest release version from Github."""
    response = requests.get(model_url)
    response.raise_for_status()
    data = response.json()
    return data['tag_name']


def download_model(model_url, destination_path):
    """Downloads the file from a Github repository"""
    with requests.get(model_url, stream=True) as response:
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def update_model_if_newer(model_url, version_file, model_name, repo, model_path):
    local_version = get_local_version(version_file)
    latest_version = get_latest_version_from_github(model_url)
    if latest_version > local_version:
        print(f"New version available: {latest_version}. Updating...")
        model_url = f"https://github.com/{repo}/releases/download/{latest_version}/{model_name}"
        download_model(model_url, model_path)
        update_local_version(latest_version, version_file)
        print(f"model updated: {latest_version}")
    else:
        print("Using the latest model version. ")
