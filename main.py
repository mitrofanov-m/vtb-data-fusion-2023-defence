import sys
from fusionlib import reliable_predict

def main():
    source_file, output_path = sys.argv[1:]
    bins_path="models/nn_bins.pickle"
    model_path="models/nn_weights.ckpt"
    result = reliable_predict(source_file, bins_path, model_path)
    result.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
