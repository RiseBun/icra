# Anonymization checklist

This review bundle was checked for:

- author names, affiliations, and email addresses;
- local usernames, hostnames, and absolute filesystem paths;
- credentials, tokens, private keys, and environment files;
- version-control history and remote URLs;
- compilation logs and document metadata that expose local paths;
- large raw datasets and private checkpoints.

The paper source retains only `Anonymous Authors`. The release directory is a
fresh tree and contains no `.git` history. Before publishing, create the remote
with an anonymous account and avoid linking it from a personal profile.
