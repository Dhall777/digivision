defmodule Digivision.Prediction_Agent do
  # import aliases + dependencies
  alias Digivision.Prediction_Agent
  alias Digivision.Data_Utils
  # alias NimbleCSV.RFC4180, as: CSV

  import Axon
  import Nx
  # import NimbleCSV

  # set module constants
  @sequence_length 1
  @sequence_features 1
  @batch_size 1
  @numbers_dataset Enum.to_list(0..100)
  @split_ratio 0.8

  def load_all_data() do
    # in a practical situation, this dataset would be pulled from a file or database
    numbers_dataset = @numbers_dataset
    split_ratio = @split_ratio

    {numbers_training_dataset_x, numbers_testing_dataset_y} = Data_Utils.dataset_split(@numbers_dataset, @split_ratio)
  end

  def load_training_dataset(numbers_training_dataset_x) do
    sequence_length = @sequence_length
    batch_size = @batch_size
    # define x_train and y_train values | perform minimal normalization
    x_train =
      numbers_training_dataset_x
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Enum.drop(-1)
      |> Nx.tensor()
      #|> Nx.divide(100)
      #|> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    y_train =
      numbers_training_dataset_x
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Enum.drop(1)
      |> Nx.tensor()
      #|> Nx.divide(100)
      #|> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    numbers_training = Stream.zip(x_train, y_train)
  end

  def load_testing_dataset(numbers_testing_dataset_y) do
    sequence_length = @sequence_length
    batch_size = @batch_size
    # define x_test and y_test values | perform minimal normalization
    x_test =
      numbers_testing_dataset_y
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Enum.drop(-1)
      |> Nx.tensor()
      #|> Nx.divide(100)
      #|> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    y_test =
      numbers_testing_dataset_y
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Enum.drop(1)
      |> Nx.tensor()
      #|> Nx.divide(100)
      #|> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    numbers_testing = Stream.zip(x_test, y_test)
  end

  def numbers_model() do
    # define numbers prediction model
    numbers_model =
      Axon.input("numbers", shape: {1, 1, 1})
      #|> Axon.dense(81, activation: :linear)
      #|> Axon.dense(70, activation: :linear)
      #|> Axon.dense(60, activation: :linear)
      #|> Axon.dense(50, activation: :linear)
      #|> Axon.dense(40, activation: :linear)
      #|> Axon.dense(30, activation: :linear)
      #|> Axon.dense(20, activation: :linear)
      |> Axon.dense(10, activation: :linear)
      |> Axon.dense(5, activation: :linear)
      |> Axon.dropout(rate: 0.00005)
      |> Axon.dense(1, activation: :linear)
  end

  def trained_model_params(numbers_model, numbers_training_dataset) do
    # train the numbers prediction model
    numbers_model_training_params =
      numbers_model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adamw(learning_rate: 0.00005), log: 50)
      |> Axon.Loop.run(numbers_training_dataset, %{}, epochs: 100, compiler: EXLA, debug: true)
  end

  def evaluate_numbers_model(numbers_model, numbers_model_training_params, numbers_testing_dataset) do
    evaluation_params =
      numbers_model
      |> Axon.Loop.evaluator()
      |> Axon.Loop.metric(:mean_absolute_error)
      #|> Axon.Loop.metric(:true_positives)
      |> Axon.Loop.metric(:accuracy)
      #|> Axon.Loop.metric(:recall)
      #|> Axon.Loop.metric(:precision)
      |> Axon.Loop.run(numbers_testing_dataset, numbers_model_training_params, iterations: 100)
  end

  def numbers_prediction(x_test, numbers_model, numbers_model_training_params) do
    sequence_length = @sequence_length
    batch_size = @batch_size
    # define input for prediction | define numbers_input via iex shell
    x_test_prep =
      x_test
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Nx.tensor()

    # predict some numbers!
    numbers_prediction =
      Axon.predict(numbers_model, numbers_model_training_params, x_test_prep, compiler: EXLA)
      |> Nx.to_flat_list()
  end

end
