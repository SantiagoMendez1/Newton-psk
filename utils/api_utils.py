import requests

headers = {'Authorization': 'Bearer 69d2d6dc'}

def request_post_data(api_url, payload):
  response = requests.post(api_url, 
                           json=payload, 
                           headers=headers)
  return response