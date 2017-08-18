try:
    from ConfigParser import ConfigParser
except ImportError:  #python3
    from configparser import ConfigParser
from os.path import dirname, join

from twilio.access_token import AccessToken, IpMessagingGrant
from twilio.rest import TwilioRestClient

# You will need your Account Sid and a API Key Sid and Secret
# to generate an Access Token for your SDK endpoint to connect to Twilio.

IDENTITY = "noreply@ecs251.com"
DEVICE_ID = "sparks-in-a-can"

with open(join(dirname(__file__), "twilio.config")) as f:
    parser = ConfigParser()
    parser.readfp(f)
    account_sid = parser.get("config", "account_sid")
    token = parser.get("config", "token")


def send_text(to, from_, body):
    client = TwilioRestClient(account_sid, token)
    client.messages.create(to=to, from_=from_, body=body)
