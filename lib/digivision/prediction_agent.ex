defmodule Digivision.Prediction_Agent do
  # import aliases + dependencies
  alias Digivision.Prediction_Agent
  import Axon
  import Nx

  def load_data() do
    sequence_length = 1
    batch_size = 1
    # define x_train and y_train values | perform minimal normalization
    x_train_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  |> Enum.chunk_every(sequence_length, 1, :discard) |> Nx.tensor() |> Nx.divide(10) |> Nx.to_batched(batch_size)
    y_train_numbers = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] |> Enum.chunk_every(sequence_length, 1, :discard) |> Nx.tensor() |> Nx.divide(10) |> Nx.to_batched(batch_size)

    numbers = Stream.zip(x_train_numbers, y_train_numbers)
  end

  def numbers_model() do
    numbers = Prediction_Agent.load_data()
    # define numbers prediction model
    numbers_model =
      Axon.input("numbers", shape: {10})
      |> Axon.dense(15, activation: :linear)
      |> Axon.dense(10, activation: :linear)
      |> Axon.dense(5, activation: :linear)
      |> Axon.dense(1, activation: :linear)
  end

  def trained_model_params() do
    # train the numbers prediction model
    numbers_model_trained =
      Prediction_Agent.numbers_model()
      |> Axon.Loop.trainer(:mean_squared_error, Axon.Optimizers.adamw(0.0005), log: 50)
      |> Axon.Loop.run(Prediction_Agent.load_data(), %{}, epochs: 40, compiler: EXLA, debug: true)
  end

  def numbers_prediction(numbers_input) do
    sequence_length = 1
    batch_size = 1
    # define input for prediction | define numbers_input via iex shell
    prediction_input =
      [numbers_input]
      |> Enum.chunk_every(sequence_length, 1, :discard)
      |> Nx.tensor()

    # predict some NUMBERS!
    numbers_prediction =
      Axon.predict(Prediction_Agent.numbers_model(), Prediction_Agent.trained_model_params, prediction_input, compiler: EXLA)
      |> Nx.to_flat_list()
  end

end
