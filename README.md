## Digivision

- Simple number prediction app built upon Axon's deep learning/DL utilities

- Project Summary:
	- This is a basic deep learning model that predicts stock and crypto prices
	- Review [this article](https://medium.com/@boring-tech-guy/elixir-axon-predicting-crypto-prices-f22d577573f4) for a full breakdown

- How to run the project:
	- start the app in Elixir's interactive shell
		- `iex -S mix`
	- establish parameters and run your predictions; run these commands sequentially:
		- `alias Digivision.Prediction_Agent`
		- `alias Digivision.Data_Utils`
		- `alias NimbleCSV.RFC4180, as: CSV`
		- `import Axon`
		- `import Nx`
		- `import NimbleCSV`
		- `{price_training_dataset, price_testing_dataset} = Prediction_Agent.load_all_data()`
		- `training_dataset = Prediction_Agent.load_training_dataset(price_training_dataset)`
		- `testing_dataset = Prediction_Agent.load_testing_dataset(price_testing_dataset)`
		- `price_model = Prediction_Agent.price_model()`
		- `price_model_training_params = Prediction_Agent.trained_model_params(price_model, training_dataset)`
		- `Prediction_Agent.evaluate_price_model(price_model, price_model_training_params, testing_dataset)`
		- Regarding `x_test` and `price_prediction`
			- `x_test` length should match sequence length (35 in this case), so include 35 prices in `x_test`
			- make sure to normalize `x_test` when making the prediction, and reverse the normlization when reading the actual predicted price

- Disclaimer
	- This app is just a manifestation of my journey with deep learning, use it as you wish.
