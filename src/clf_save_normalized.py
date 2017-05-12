import classifier


def main():
    print('loading raw/means')
    X, y = classifier.load_data('raw/means')

    #X_n = classifier.normalize(X, mode='each')
    #classifier.save_to_file(X_n, y, 'normalized_means_by_each.csv', 'normalized/means')

    print('normalizing raw/means')
    X_n, max_peak = classifier.normalize(X, mode='total')
    print('saving normalized data from raw/means')
    classifier.save_to_file(X_n, y, 'normalized_means_by_total.csv', 'normalized/means')

    print('loading raw/members')
    X, y = classifier.load_data('raw/members')

    print('normalizing raw/members')
    X_n, max_peak = classifier.normalize(X, mode='total', max_peak=max_peak)
    #classifier.save_to_file(X_n, y, 'normalized_members_by_each.csv', 'normalized/members')

    print('saving normalized data from raw/members')
    classifier.save_to_file(X_n, y, 'normalized_members_by_total.csv', 'normalized/members')

    return 0


if __name__ == '__main__':
    main()
