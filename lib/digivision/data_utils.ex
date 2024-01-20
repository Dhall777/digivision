defmodule Digivision.Data_Utils do
  # import aliases + dependencies
  # alias alias Digivision.Prediction_Agent
  alias Nx.Defn
  # numerical defs + ml utilities
  import Nx.Defn

  # split the data into a training set and testing/validation set (x_training_dataset and y_testing_dataset)
  def dataset_split(numbers_dataset, split_ratio) when split_ratio > 0.0 and split_ratio < 1.0 do
    dataset_count = Enum.count(numbers_dataset)
    train_count = round(split_ratio * dataset_count)
    # after some prep, finally split the data
    {Enum.take(numbers_dataset, train_count), Enum.drop(numbers_dataset, train_count)}
  end

end
