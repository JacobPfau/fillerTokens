import argparse

from src.match2 import STRING_FUNCTION_MAPPING, GenerateMatch2Dataset

def main():
    parser = argparse.ArgumentParser(description='Generate Match2 String CSVs and hyper parameter json')
    
    parser.add_argument('--name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--train_samples', type=float, default=1e7, help='Number of training samples, to convert to int')
    parser.add_argument('--test_samples', type=float, default=2e3, help='Number of test samples, to convert to int')

    parser.add_argument('--dimension', type=int, default=4, help='Dimension for Match2 class')
    parser.add_argument('--mod', type=int, default=10, help='Mod value')
    parser.add_argument('--length', type=int, default=10, help='Length of data')
    parser.add_argument('--true_instance_rate', type=float, default=0.1, help='True instance rate')
    parser.add_argument('--cot_rate', type=float, default=0.5, help='cot rate')
    parser.add_argument('--no_filler_rate', type=float, default=0, help='No filler rate')
    parser.add_argument('--corruption_rate', type=float, default=4/3, help='Corruption rate for False instances, Geometric with this rate')
    parser.add_argument('--transform', type=str, choices=['identity', 'lookup'], default='lookup', help='Type of transformation')
    
    parser.add_argument('--filler_to_string', type=str, choices=['b10_repeat',], default='b10_repeat', help='Function for filler string transformation')
    parser.add_argument('--cot_to_string', type=str, choices=['b10_basic'], default='b10_basic', help='Function for cot string transformation')
    parser.add_argument('--no_filler_to_string', type=str, choices=['b10_no_filler',], default='b10_no_filler', help='Function for no filler string transformation')

    parser.add_argument('--data_path', type=str, default='./data/', help='Path to save the dataset')

    args = parser.parse_args()
    
    GenerateMatch2Dataset(
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
        transform=args.transform,
        filler_to_string=STRING_FUNCTION_MAPPING[args.filler_to_string],
        cot_to_string=STRING_FUNCTION_MAPPING[args.cot_to_string],
        no_filler_to_string=STRING_FUNCTION_MAPPING[args.no_filler_to_string],
        data_path=args.data_path
    )
    return

if __name__ == '__main__':
    main()
