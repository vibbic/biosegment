BioSegment uses [GitHub Actions](https://github.com/vibbic/biosegment/actions) to automate testing, linting and documentation updates. The configuration files are located in `.github/`.

## Documentation

The workflow `.github/workflows/ci.yml` rebuilds the documentation site when the `main` branch updates. The resulting site is published on the `gh-pages` branch. More documentation on publishing can be found [here](https://squidfunk.github.io/mkdocs-material/publishing-your-site/).

## Linting

## Testing

## Building images


## Local development
For local development, [act](https://github.com/nektos/act) can be used.

```bash
# install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

Example commands:
```bash
act -P ubuntu-latest=berombau/act_base -j test
# WARNING act-environments-ubuntu:18.04 is >18GB!
act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04 -j lint
```

Caveats

- There are differences between `berombau/act_base` and the GitHub images
- different oses are not supported
- public GitHub actions can depend on certain GitHub tooling, which would require incorporating that dependency in the `act_base` image.
