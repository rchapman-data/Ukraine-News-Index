# AWS's own base image for Python Lambda functions. Amazon Linux underneath,
# with the Lambda Runtime Interface Client already built in -- that's the
# piece that lets Lambda actually call the function inside this container
# (see CMD at the bottom); it's why we start from this instead of a plain
# python:3.12 image.
FROM public.ecr.aws/lambda/python:3.12

# Where the embedding model gets cached when we download it below. This is
# NOT /tmp deliberately: Lambda gives every real invocation a fresh, empty
# /tmp, so anything baked into /tmp during this build would be invisible at
# runtime. Everywhere else in the image (like this path) is ordinary image
# content, so it survives.
ENV SENTENCE_TRANSFORMERS_HOME=/var/task/model_cache

# Dependencies in their own layer, before the application code. Docker
# rebuilds a layer, and everything after it, only when something in that
# layer actually changes -- keeping requirements.txt separate from the code
# means editing lambda_handler.py later won't force a slow full reinstall
# of PyTorch and everything else, only the layers from the code copy down.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model's weights into the image at build time (see
# build-progress.md, Part 6, for the full reasoning) -- this line downloads
# the model once, now, while building, and genuinely needs real internet
# access to do that (unlike everything from here on).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Force the library to only ever use the files just baked in above, never a
# network call to Hugging Face. Without this, sentence-transformers still
# makes a lightweight "any updates?" request to the Hub every time the model
# loads -- harmless during local testing (which has ordinary internet), but
# a real Lambda may have no internet access at all, so that same request
# could hang or fail instead of just being skipped. Deliberately set only
# after the download above, so the download itself isn't blocked by it.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY lambda_handler.py .

# Tells Lambda which function to call: <file name without .py>.<function name>
CMD ["lambda_handler.lambda_handler"]
