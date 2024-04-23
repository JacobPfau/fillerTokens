from src import GenerateMatch3Dataset
from src.match3 import STRING_FUNCTION_MAP
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate Match3 String CSVs and hyper parameter json')
    
    parser.add_argument('--name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--train_samples', type=float, default=1e7, help='Number of training samples, to convert to int')
    parser.add_argument('--test_samples', type=float, default=2e3, help='Number of test samples, to convert to int')

    parser.add_argument('--dimension', type=int, default=3, help='Dimension for Match3 class')
    parser.add_argument('--mod', type=int, default=10, help='Mod value')
    parser.add_argument('--length', type=int, default=10, help='Length of data')
    parser.add_argument('--true_instance_rate', type=float, default=0.5, help='True instance dataset percentage')
    parser.add_argument('--cot_rate', type=float, default=0.5, help='cot dataset percentage')
    parser.add_argument('--no_filler_rate', type=float, default=0, help='No intermediate tokens dataset percentage')
    parser.add_argument('--corruption_rate', type=float, default=4/3, help='Corruption dataset percentage for False instances, Geometric with this rate')
    
    parser.add_argument('--cot_to_string', type=str, choices=['rand_cot', 'serial'], default='rand_cot', help='Function for cot string transformation')

    parser.add_argument('--data_path', type=str, default='/scratch/jp6263/slackV2/data/', help='Path to save the dataset')

    args = parser.parse_args()

    GenerateMatch3Dataset(
        name=args.name,
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
        dimension=args.dimension,
        mod=args.mod,
        length=args.length,
        true_instance_rate=args.true_instance_rate,
        cot_rate=args.cot_rate,
        no_filler_rate=args.no_filler_rate,
        corruption_rate=args.corruption_rate,
        cot_to_string=STRING_FUNCTION_MAP[args.cot_to_string],
        data_path=args.data_path
    )
    return

if __name__ == '__main__':
    main()