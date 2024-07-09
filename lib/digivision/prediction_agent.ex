defmodule Digivision.Prediction_Agent do
  # import aliases + dependencies
  alias Digivision.Prediction_Agent
  alias Digivision.Data_Utils
  alias NimbleCSV.RFC4180, as: CSV

  import Axon
  import Nx
  import NimbleCSV

  # set module constants
  @sequence_length 30
  @sequence_features 2
  @batch_size 14
  @price_dataset "/usr/local/elixir-apps/digivision/priv/TSLA/TSLA.csv" |> File.stream!() |> CSV.parse_stream() |> Stream.map(fn [date, close] -> [Integer.parse(date) |> elem(0), Float.parse(close) |> elem(0)] end) |> Enum.chunk_every(@sequence_length, @sequence_length, :discard)
  @split_ratio 0.8

  def load_all_data() do
    # in a practical situation, this dataset would be pulled from a file or database
    price_dataset = @price_dataset
    split_ratio = @split_ratio

    {price_training_dataset, price_testing_dataset} = Data_Utils.dataset_split(@price_dataset, @split_ratio)
  end

  def load_training_dataset(price_training_dataset) do
    sequence_length = @sequence_length
    batch_size = @batch_size
    # define x_train and y_train values | perform minimal normalization
    x_train =
      price_training_dataset
      |> Enum.drop(-1)
      |> Nx.tensor()
      |> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    y_train =
      price_training_dataset
      |> Enum.drop(1)
      |> Nx.tensor()
      |> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    price_training_zipped = Stream.zip(x_train, y_train)
  end

  def load_testing_dataset(price_testing_dataset) do
    sequence_length = @sequence_length
    batch_size = @batch_size
    # define x_test and y_test values | perform minimal normalization
    x_test =
      price_testing_dataset
      |> Enum.drop(-1)
      |> Nx.tensor()
      |> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    y_test =
      price_testing_dataset
      |> Enum.drop(1)
      |> Nx.tensor()
      |> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    price_testing_zipped = Stream.zip(x_test, y_test)
  end

  def price_model() do
    # define price prediction model
    price_model =
      Axon.input("prices", shape: {nil, @sequence_length, @sequence_features})
      |> Axon.dense(200, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(190, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(180, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(170, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(160, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(150, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(140, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(130, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(120, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(110, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(100, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(90, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(80, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(70, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(60, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(50, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(40, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(30, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(20, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(10, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dense(5, kernel_initializer: :he_uniform, activation: :relu)
      |> Axon.dropout(rate: 0.05)
      |> Axon.dense(2, kernel_initializer: :he_uniform, activation: :relu)  end

  def trained_model_params(price_model, price_training_dataset) do
    # train the price prediction model
    price_model_training_params =
      price_model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adamw(learning_rate: 0.00005), log: 50)
      |> Axon.Loop.run(price_training_dataset, %{}, epochs: 100, compiler: EXLA, debug: true)
  end

  def evaluate_price_model(price_model, price_model_training_params, price_testing_dataset) do
    evaluation_params =
      price_model
      |> Axon.Loop.evaluator()
      |> Axon.Loop.metric(:mean_absolute_error)
      #|> Axon.Loop.metric(:true_positives)
      #|> Axon.Loop.metric(:accuracy)
      #|> Axon.Loop.metric(:recall)
      #|> Axon.Loop.metric(:precision)
      |> Axon.Loop.run(price_testing_dataset, price_model_training_params, iterations: 100)
  end

  def price_prediction(x_test, price_model, price_model_training_params) do
    sequence_length = 1
    sequence_features = 2
    # define input for prediction | define price_input via iex shell
    x_test_prep =
      x_test
      |> Enum.chunk_every(sequence_length, sequence_length, :discard)
      |> Nx.tensor()
      |> Nx.reshape({:auto, sequence_length, sequence_features})

    # predict some prices!
    price_prediction =
      Axon.predict(price_model, price_model_training_params, x_test_prep, compiler: EXLA)
      |> Nx.to_flat_list()
  end

end
