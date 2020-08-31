# from requests_oauthlib import OAuth2Session
import json
import requests
from requests_oauthlib import OAuth2Session

# TODO remove credentials and use client secret
ROOT_USERNAME = "admin@biosegment.irc.ugent.be"
ROOT_PASSWORD = "m1cr0scopy"
# CLIENT_ID = "<your client key>"
# CLIENT_SECRET = "<your client secret>"

class Base(object):
    __instance = None

    API_ROOT = "http://localhost/api/v1/"
    TOKEN_URL = f'{API_ROOT}login/access-token'
    tokens = None

    def __init__(self):
        """ Virtually private constructor. """
        if Base.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Base.__instance = self
        # get tokens for the first time
        self.get_tokens()

    @classmethod
    def get_tokens(cls):
        if Base.tokens is not None:
            return Base.tokens["access_token"]
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

        access_token_response = requests.post(Base.TOKEN_URL, data=data, verify=False, allow_redirects=False, 
        # auth=(client_id, client_secret)
        )

        # print(access_token_response.headers)
        # print(access_token_response.text)

        tokens = json.loads(access_token_response.text)
        # print("access token: " + tokens['access_token'])
        Base.tokens = tokens
        # print(Base.tokens)
        token = tokens['access_token']
        return token
    
    @classmethod
    def get(cls, path, headers={}, **kwargs):
        if cls.tokens is None:
            cls.get_tokens()
        r = requests.get(
            f"{cls.API_ROOT}{path}", 
            headers={
                'Authorization': 'Bearer ' + cls.tokens['access_token'], 
                # **headers
            }, 
            **kwargs
        )
        assert r.status_code == 200
        return r.json()
