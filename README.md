## Digivision

- Simple number prediction app built upon Axon's deep learning/DL utilities

- Project Summary:
	- This is a basic deep learning model that predicts the next number in a sequence using only dense layers.
	- It's sorta like teaching someone to count from 0-19, giving them a random number, then asking them what number comes after the random number you just gave them.
	- The model performs well when you "ask" it to predict numbers between 1-99; after that, the predictions become a bit less accurate
		- this behavior is completely expected since we only train the model on numbers between 0-19
		- so, theoretically, if you just include more numbers in the training dataset (e.g., 0-100000), the more accurate its predictions will become

- How to run the project:
	- start the app in Elixir's interactive shell
		- `iex -S mix`
	- run your prediction
		- `Digivision.Prediction_Agent.numbers_prediction(numbers_input)`
		- `numbers_input` expects an integer; I should get used to defining this via Elixir's typespecs annotation... meh.

- Disclaimer:
	- The model starts to predict the next number in a sequence with nearly 100% accuracy after a few training iterations.
		- Why would it take several iterations before getting predictions correct? Idk yet but it's pretty cool when it gets it right.
	- This app is just a manifestation of my journey with deep learning, use it as you wish.
