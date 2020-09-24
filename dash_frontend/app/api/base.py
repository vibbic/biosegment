# from requests_oauthlib import OAuth2Session
import http.client
import json
import logging

from requests.packages.urllib3.util.retry import Retry

from app.api.TimeoutHTTPAdapter import TimeoutHTTPAdapter
# from requests_oauthlib import OAuth2Session
from app.env import API_DOMAIN
from requests_toolbelt import sessions

# TODO remove credentials and use client secret
ROOT_USERNAME = "admin@biosegment.irc.ugent.be"
ROOT_PASSWORD = "m1cr0scopy"
# CLIENT_ID = "<your client key>"
# CLIENT_SECRET = "<your client secret>"

API_ROOT = f"http://{API_DOMAIN}/api/v1/"
TOKEN_URL = "login/access-token"

http_session = sessions.BaseUrlSession(base_url=API_ROOT)

retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
# Mount it for both http and https usage
adapter = TimeoutHTTPAdapter(max_retries=retries)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)

# setup logging
http.client.HTTPConnection.debuglevel = 1

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True


def get_tokens():
    global http_session
    # TODO refresh tokens
    data = {
        "grant_type": "password",
        "username": ROOT_USERNAME,
        "password": ROOT_PASSWORD,
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
        access_token_response = http_session.post(
            TOKEN_URL,
            data=data,
            timeout=5,
            # auth=(client_id, client_secret)
        )
        logging.debug(access_token_response)
        # print(access_token_response.headers)
        # print(access_token_response.text)
        new_tokens = json.loads(access_token_response.text)
        # print("access token: " + tokens['access_token'])
        tokens = new_tokens
        # print(Base.tokens)
        token = tokens["access_token"]
        logging.debug(f"Token type: {type(token)}")
    except Exception as e:
        logging.warning(f"Exception {e}")
        return None
    return str(token)


token = get_tokens()
while token is None:
    token = get_tokens()


def get(path, headers={}, **kwargs):
    global token
    global http_session
    if not token:
        token = get_tokens()
    # logging.debug(f"GET using token {token}")
    # logging.debug(f"Path {path}")
    r = http_session.get(
        path,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
            # **headers
        },
        timeout=1,
    )
    assert r.status_code == 200
    # logging.debug(f"f {r}")
    return r.json()


def post(path, headers={}, **kwargs):
    global token
    global http_session
    if not token:
        token = get_tokens()
    # logging.debug(f"POST using token {token}")
    # logging.debug(f"Path {path}")
    try:
        payload = kwargs["json"]
    except:
        payload = None
    # logging.debug(f"JSON {payload}")
    r = http_session.post(
        path,
        headers={
            "Authorization": "Bearer " + token,
            # **headers
            "Content-type": "application/json",
            "Accept": "application/json",
        },
        timeout=1,
        json=payload,
    )
    assert r.status_code == 200 or r.status_code == 201
    # logging.debug(f"f {r}")
    return r.json()
