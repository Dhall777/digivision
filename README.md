## Digivision

- Simple number prediction app built upon Axon's deep learning/DL utilities

- Project Summary:
	- This is a basic deep learning model that is used to predict the next number in a sequence. 
	- It's sorta like teaching someone to count from 0-19, giving them a random number, then hoping they generalize that 'counting' knowledge to tell you the number that succeeds the given random number.

- How to run the project:
	- start the app in Elixir's interactive shell
		- iex -S mix
	- run your prediction
		- Digivision.Prediction_Agent.numbers_prediction(numbers_input)
		- numbers input expects an integer; I should get used to defining this via Elixir's typespecs annotation... meh.

- Disclaimer:
	- The model starts to predict the next number in a sequence with nearly 100% accuracy after a few training processes. 
	- Why would it take several processes before getting predictions correct? Idk yet but it's pretty cool when it gets it right.
	- This is just a side project, so I wouldn't advise using this in production environments.
