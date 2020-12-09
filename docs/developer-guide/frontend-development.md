# Frontend development

- Install locally [Node](https://nodejs.org/en/download/) and npm
- Enter the `frontend` directory, install the NPM packages and start the live server using the `npm` scripts:

```bash
cd frontend
npm install
npm run serve
```

Then open your browser at http://localhost:8080

Notice that this live server is not running inside Docker, it is for local development, and that is the recommended workflow. Once you are happy with your frontend, you can build the frontend Docker image and start it, to test it in a production-like environment. But compiling the image at every change will not be as productive as running the local development server with live reload.

Check the file `package.json` to see other available options.

```bash
# unit test
npm run unit:test
# lint
npm run lint
```

If you have Vue CLI installed, you can also run `vue ui` to control, configure, serve, and analyze your application using a nice local web user interface.

## Structure

- `frontend/src/`
    - `api/`
        - code generated backend API
    - `components/`
        - Vue components that use vuex store modules
    - `router/`
        - routes views using vue-router
    - `views/`
        - Uses components to create a webpage
    - `store/`
        - Vuex store modules that contain state. Actions use the API.

## Vue

The frontend is implemented in Vue. Read up on Vue, Typescript, vue-router, vuex, axios and Jest.

## Vuetify

The frontend uses a Vue UI library named [Vuetify](https://vuetifyjs.com/en/). It features a list of ready-made components, see their documentation for more information.

## Adding a new feature

- Regenerate the backend API using [code generation](/developer-guide/code-generation).
- Use new API feature by editing a module in the Vuex `/store` or creating a new one.
- create a new view in `/views` and add it to the `/router`.
- create a new component in `/components` that uses the new vuex module.
- add the new component to the new view.
- write a Jest test for the new code.
- Check if there are no Typescript errors.
- Lint.