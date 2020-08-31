# from requests_oauthlib import OAuth2Session
import json
import requests
# from requests_oauthlib import OAuth2Session

# TODO remove credentials and use client secret
ROOT_USERNAME = "admin@biosegment.irc.ugent.be"
ROOT_PASSWORD = "m1cr0scopy"
# CLIENT_ID = "<your client key>"
# CLIENT_SECRET = "<your client secret>"
API_ROOT = "http://localhost/api/v1/"
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
    try:
        access_token_response = requests.post(TOKEN_URL, data=data, timeout=0.02,
        # auth=(client_id, client_secret)
        )
        # print(access_token_response.headers)
        # print(access_token_response.text)
        new_tokens = json.loads(access_token_response.text)
        # print("access token: " + tokens['access_token'])
        tokens = new_tokens
        # print(Base.tokens)
        token = tokens['access_token']
    except:
        token = None
    return token
    
def get(path, token, headers={}, **kwargs):
    assert token is not None
    r = requests.get(
        f"{API_ROOT}{path}", 
        headers={
            'Authorization': 'Bearer ' + token, 
            # **headers
        },
        timeout=0.001,
        **kwargs
    )
    assert r.status_code == 200
    return r.json()
