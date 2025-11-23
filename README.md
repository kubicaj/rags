# rags

Repository containing various RAGs implementations

## Environment variables

Here you can see the environment variables used in this repository. You can set them in a `.env` file or directly in
your environment.

- `OPENAI_API_KEY`: Your OpenAI API key for accessing OpenAI services.
- `AWS_ACCESS_KEY_ID`: Your AWS access key ID for accessing AWS services.
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key for accessing AWS services.
- `AWS_REGION`: The AWS region where your services are hosted (e.g., `us-east-1`).

## Support tools for this project

In this section, we outline the support tools used in this project to enhance development and maintain code quality.

### UV

This project uses [UV](https://docs.astral.sh/uv/) installer to manage dependencies and virtual environments. To install
UV, you can use pip:

```bash
pip install uv
```

Another way how to install you can find in
the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

Please install the UV out of the virtual environment to manage it properly.

### Ruff linter

This project uses [Ruff](https://docs.astral.sh/ruff/) as a linter to ensure code quality and consistency. To install
Ruff, you can use pip:

```bash
pip install ruff
```

### Github Actions

This project utilizes GitHub Actions for continuous integration. The workflows are defined in the
`.github/workflows` directory. These workflows automate tasks such as running tests, linting code

Note: Deployment not used yet.

## Possible enhancements

- Add batch processing of embeddings for improved performance.
  See [OpenAI official documentation for more details](https://platform.openai.com/docs/guides/batch)
- Add Settings management for better configuration handling and management.
