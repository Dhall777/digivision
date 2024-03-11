## Digivision

- Simple number prediction app built upon Axon's deep learning/DL utilities

- Project Summary:
	- This is a basic deep learning model that predicts the next number in a sequence using only dense layers.
	- Review [this article](https://medium.com/@boring-it-guy/elixir-nx-axon-building-a-number-prediction-model-776c0dfe97b0) for a full breakdown

- How to run the project:
	- start the app in Elixir's interactive shell
		- `iex -S mix`
	- establish parameters and run your predictions; run these commands sequentially
		- `alias Digivision.Prediction_Agent`
		- `alias Digivision.Data_Utils`
		- `import Axon`
		- `import Nx`
		- `{numbers_training_dataset, numbers_testing_dataset} = Prediction_Agent.load_all_data()`
		- `training_dataset = Prediction_Agent.load_training_dataset(numbers_training_dataset)`
		- `testing_dataset = Prediction_Agent.load_testing_dataset(numbers_testing_dataset)`
		- `numbers_model = Prediction_Agent.numbers_model()`
		- `numbers_model_training_params = Prediction_Agent.trained_model_params(numbers_model, training_dataset)`
		- `Prediction_Agent.evaluate_numbers_model(numbers_model, numbers_model_training_params, testing_dataset)`
		- `x_test = [80]`
		- `Prediction_Agent.numbers_prediction(x_test, numbers_model, numbers_model_training_params)`
		- `numbers_input` expects an integer (I should get used to defining this via Elixir's typespecs annotation... meh)
		- `x_test` can be whatever number your heart desires

- Disclaimer
	- This app is just a manifestation of my journey with deep learning, use it as you wish.
