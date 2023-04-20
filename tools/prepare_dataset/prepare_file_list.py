from common import *


def get_file_list(
        data_dir,
        subsets=('train', 'val', 'test'),
        subdirs=('A', 'B', 'label'),
        glob_pattern='*',
        file_list_pattern="{subset}.txt",
        abs_path=False,
        sep=' ', ):
    '''
    Get file list for the cropped result
    Args:
        data_dir (str): the directory for the generation of the file list
        subsets (tuple|list|None, optional): List or tuple of names of subsets
            or None. Images to be cropped should be stored in `data_dir/subset/subdir/`
            or `data_dir/subdir/` (when `subsets` is set to None), where `subset` is an
            element of `subsets`. Defaults to ('train', 'val', 'test').
        subdirs (tuple|list, optional): List or tuple of names of subdirectories. Images
            to be cropped should be stored in `data_dir/subset/subdir/` or
            `data_dir/subdir/` (when `subsets` is set to None), where `subdir` is an
            element of `subdirs`. Defaults to ('A', 'B', 'label').
        glob_pattern (str, optional): Glob pattern used to match image files.
            Defaults to '*', which matches arbitrary file.
        file_list_pattern(str, optional): Filt list pattern used to generate file list.
            Defaults to '{subset}.txt', which matches subset in subsets and generate txt file.
        abs_path (bool, optional):  Whether to store the absolute path in file list.
            Defaults to 'False', which indicates the relative path is saved in the file list.
        sep (str, optional): Delimiter to use when writing lines to file list.
            Defaults to ' '.
    '''
    for subset in subsets:
        path_tuples = get_path_tuples(
            *(osp.join(data_dir, subset, subdir) for subdir in subdirs),
            glob_pattern=glob_pattern,
            data_dir=data_dir)
        if abs_path:
            pass
            path_tuples_new = []
            for path_tuple in path_tuples:
                path_tuple_new = [
                    os.path.join(data_dir, path_t) for path_t in path_tuple
                ]
                path_tuples_new.append(tuple(path_tuple_new))
            path_tuples = path_tuples_new

        file_list = osp.join(data_dir, file_list_pattern.format(subset=subset))
        create_file_list(file_list, path_tuples, sep)
        print(f"Write file list to {file_list}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        help="the directory for the generation of the file list")
    parser.add_argument(
        '--subsets',
        nargs="*",
        default=['train', 'val', 'test'],
        help="List or tuple of names of subsets", )
    parser.add_argument(
        '--subdirs',
        nargs="*",
        default=['A', 'B', 'label'],
        help="List or tuple of names of subdirectories of subsets", )
    parser.add_argument(
        '--glob_pattern',
        type=str,
        default='*',
        help="Glob pattern used to match image files", )
    parser.add_argument(
        '--file_list_pattern',
        type=str,
        default='{subset}.txt',
        help="Filt list pattern used to generate file list", )
    parser.add_argument(
        '--abs_path',
        action='store_true',
        help='Whether to store the absolute path in file list', )
    parser.add_argument(
        '--sep',
        type=str,
        default=' ',
        help="Delimiter to use when writing lines to file list", )
    args = parser.parse_args()
    get_file_list(
        data_dir=args.data_dir,
        subsets=args.subsets,
        subdirs=args.subdirs,
        glob_pattern=args.glob_pattern,
        file_list_pattern=args.file_list_pattern,
        abs_path=args.abs_path,
        sep=args.sep, )
