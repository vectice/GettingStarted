# Vectice Getting started
Public repository with Vectice's samples, tutorial notebooks and data

A tutorial for using Vectice.com. The notebooks and data are to be used in conjuntion with an existing project in Vectice (Forecast in-store unit sales).

The Vectice API documentation can be found [here](https://docs.vectice.com/)

## Getting Started
Besides the required packages needed to run the notebooks, the Vectice package also needs to be installed:
```
pip install vectice
```

The following code is an example to test that the Vectice package is properly installed and works as it should. This line tries to establish a default connection to Vectice, testing the connection and not authentication:
```
import vectice
my_vectice = vectice.connect()
```
The command above should generate a warning:
```
You must provide the api_token. You can generate them by going to the page https://app.vectice.com/account/api-keys
```
Vectice supports different ways of connecting through the API, you can pass the connect() method a configuration file or specify the values in the method argument:
```
my_project = vectice.connect(
        api_token=USER API TOKEN,
        host="https://app.vectice.com",
        workspace="Project Workspace",
        project="My Test Project",
    )
```
The code above would return a Project object, omiting the project would return a Workspace object and omitting both would return a Connection object.

```
retail_ws = vectice.connect(config=r"API_token.json")
```
The connection above uses a configuration file. This is the preferred method of connecting as it provides better security of your API token. The json file has this format:
```
{
  "VECTICE_API_ENDPOINT": VECTICE_URL,
  "VECTICE_API_TOKEN": YOUR API KEY,
  "WORKSPACE": "PROJECT WORKSPACE"
}
```
As previously stated, this connection would return a Workspace object.

