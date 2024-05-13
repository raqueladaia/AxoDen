# How to Create a Release

- Make sure tests and linting are green


# Create a Release on github and PyPi
Releases are automated with github actions.
They run whenever a new tag `v*` is created.

1. Update the version in `axoden/__init__.py`.
2. push and make sure linting and tests are green.
3. Create a new tag and push it to the repo:
    - `git tag vx.y.z`, this should match the version in `axoden/__init__.py`
    - `git push --tags`
4. This will start three github actions:
    - `publish.yml` will build the pip package using flit and push it to pypi.org
    - `build_executables.yml` will build standalone executables for windows, macos and ubuntu. It will also create a release from the new tag and add those executables (in .zip format) to the release so they can be downloaded.
    - `docs.yml` will build the documentation using sphinx and update the branch `docs`. This will update the docs on github pages.

# Releasing Streamlit App
The streamlit branch is automatically deployed from the `release` branch.
Simply merge the changes from `main` to `release`:

```bash
git checkout release
git merge main
git push
```