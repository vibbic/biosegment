# from requests_oauthlib import OAuth2Session
import json
import requests
import logging
# from requests_oauthlib import OAuth2Session
from app.env import API_DOMAIN

# TODO remove credentials and use client secret
ROOT_USERNAME = "admin@biosegment.irc.ugent.be"
ROOT_PASSWORD = "m1cr0scopy"
# CLIENT_ID = "<your client key>"
# CLIENT_SECRET = "<your client secret>"

API_ROOT = f"http://{API_DOMAIN}/api/v1/"
TOKEN_URL = f'{API_ROOT}login/access-token'

def get_tokens():
    # TODO refresh tokens
    data = {
        'grant_type': 'password',
        'username': ROOT_USERNAME, 
        'password': ROOT_PASSWORD,
        # 'scope': '',
        # 'cient_id': '',
        # 'client_secret': '',
    }

    # token = oauth.fetch_token(
    #         TOKEN_URL,
    #         authorization_response=authorization_response,
    #         # Google specific extra parameter used for client
    #         # authentication
    #         # client_secret=client_secret
    #         )

    # r = oath.get(f"{API_ROOT}projects")
    logging.debug(data)
    logging.debug(TOKEN_URL)
    try:
        access_token_response = requests.post(TOKEN_URL, data=data, 
        timeout=1,
        # auth=(client_id, client_secret)
        )
        logging.debug(access_token_response)
        # print(access_token_response.headers)
        # print(access_token_response.text)
        new_tokens = json.loads(access_token_response.text)
        # print("access token: " + tokens['access_token'])
        tokens = new_tokens
        # print(Base.tokens)
        token = tokens['access_token']
        logging.debug(token)
    except Exception as e:
        logging.debug(f"Error {e}")
        token = None
    return token
    
def get(path, token, headers={}, **kwargs):
    assert token is not None
    logging.debug("GET using token")
    logging.debug(f"Path {path}")
    r = requests.get(
        f"{API_ROOT}{path}", 
        headers={
            'Authorization': 'Bearer ' + token, 
            # **headers
        },
        timeout=1,
        **kwargs
    )
    assert r.status_code == 200
    logging.debug(R"f {r}")
    return r.json()
