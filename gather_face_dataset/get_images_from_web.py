# Microoft asure account
# Key 1: 8f119e5b4a5d42b98dceeff664c27f8a
# Key 2: 5fec291aa84245409a693f6daeab078b
# Endpoint: https://westcentralus.api.cognitive.microsoft.com/face

# import the necessary packages
from requests import exceptions
import argparse
import requests
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
                help="search query to search Bing Image API for")
ap.add_argument("-o", "--face_recognizing_model", required=True,
                help="path to face_recognizing_model directory of images")
args = vars(ap.parse_args())

API_KEY = "8f119e5b4a5d42b98dceeff664c27f8a"
MAX_RESULTS = 50
GROUP_SIZE = 10
# set the endpoint API URL
URL = "https://westcentralus.api.cognitive.microsoft.com/face"

# when attempting to download images from the web both the Python
# programming language and the requests library have a number of
# exceptions that can be thrown so let's build a list of them now
# so we can filter on them
EXCEPTIONS = {IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError,
              exceptions.Timeout}

# headers and search parameters
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# make the search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults, term))
total = 0

# loop over the results
for v in results["value"]:
    try:
        # make a request to download the image
        print("[INFO] fetching: {}".format(v["contentUrl"]))
        r = requests.get(v["contentUrl"], timeout=30)

        # build the path to the face_recognizing_model image
        ext = v["contentUrl"][v["contentUrl"].rfind("."):]
        p = os.path.sep.join([args["face_recognizing_model"], "{}{}".format(
            str(total).zfill(8), ext)])

        f = open(p, "wb")
        f.write(r.content)
        f.close()
    except Exception as e:
        if type(e) in EXCEPTIONS:
            print("[INFO] skipping: {}".format(v["contentUrl"]))
            continue

    # try to load the image from disk
    image = cv2.imread(p)
    # if the image is `None` then we could not properly load the
    # image from disk (so it should be ignored)
    if image is None:
        print("[INFO] deleting: {}".format(p))
        os.remove(p)
        continue
    # update the counter
    total += 1