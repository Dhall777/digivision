defmodule Digivision.Prediction_Agent do
  # import aliases + dependencies
  alias Digivision.Prediction_Agent
  alias Digivision.Data_Utils
  alias NimbleCSV.RFC4180, as: CSV

  import Axon
  import Nx
  import NimbleCSV
  import Polaris

  # set module constants
  @sequence_length 35
  @sequence_features 1
  @batch_size 5
  @split_ratio 0.8
  @eth_price_dataset "/usr/local/elixir-apps/digivision/priv/ETH-USD/ETH-USD.csv" 

  def load_all_data() do
    eth_data = @eth_price_dataset
      |> File.stream!()
      |> CSV.parse_stream()
      |> Stream.map(fn [date, _open, _high, _low, close, _adj_close, _volume] -> {Date.from_iso8601!(date), String.to_float(close)} end)
      |> Enum.map(fn {_date, close} -> close end)
      |> Enum.chunk_every(@sequence_length, @sequence_length, :discard)
    # Load & split dataset into training and testing sets
    {price_training_dataset, price_testing_dataset} = Data_Utils.dataset_split(eth_data, @split_ratio)
  end

  def load_training_dataset(price_training_dataset) do
    sequence_length = @sequence_length
    batch_size = @batch_size
    # define x_train and y_train values | fixed normalization method is not recommended, but it works for this article (essentially is oversimplified MinMax)
    x_train =
      price_training_dataset
      |> Enum.drop(-1)
      |> Nx.tensor()
      |> Nx.divide(10000)
      |> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    y_train =
      price_training_dataset
      |> Enum.drop(1)
      |> Nx.tensor()
      |> Nx.divide(10000)
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
      |> Nx.divide(10000)
      |> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    y_test =
      price_testing_dataset
      |> Enum.drop(1)
      |> Nx.tensor()
      |> Nx.divide(10000)
      |> Nx.reshape({:auto, @sequence_length, @sequence_features})
      |> Nx.to_batched(@batch_size)

    price_testing_zipped = Stream.zip(x_test, y_test)
  end

  def price_model() do
    # define price prediction model
    price_model =
      Axon.input("prices", shape: {nil, @sequence_length, @sequence_features})
      |> Axon.lstm(50, activation: :relu)
      |> then(fn {output, _} -> output end)
      |> Axon.lstm(50, activation: :relu)
      |> then(fn {output, _} -> output end)
#      |> Axon.conv(200, padding: :same, activation: :relu)
#      |> Axon.max_pool(padding: :same)
      |> Axon.dense(50, activation: :relu)
      |> Axon.dense(25, activation: :relu)
      |> Axon.dense(@sequence_features)
  end

  def trained_model_params(price_model, price_training_dataset) do
    # train the price prediction model
    price_model_training_params =
      price_model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 0.001), log: 50)
      |> Axon.Loop.run(price_training_dataset, %{}, epochs: 3, compiler: EXLA, debug: true)
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
      |> Axon.Loop.run(price_testing_dataset, price_model_training_params, compiler: EXLA, iterations: 100)
  end

  def price_prediction(x_test, price_model, price_model_training_params) do
    sequence_length = 35
    sequence_features = 1
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
      |> List.first()
  end

end
