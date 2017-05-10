import classifier


def main():
    X_means, y_means = classifier.load_data('means')
    ########################################################################       
    # Normalizing                                                               
    ########################################################################       

    #X_means_n1, max_peak = classifier.normalize(X_means, mode='total')
    X_means_n2 = classifier.normalize(X_means, mode='each')

    #classifier.save_to_file(X_means_n1, y_means, 'normalized_means_by_total.csv', 'normalized/means')
    classifier.save_to_file(X_means_n2, y_means, 'normalized_means_by_each.csv', 'normalized/means')

    X_members, y_members = classifier.load_data('members')
    #X_members_n1, = classifier.normalize(X_members, mode='total', max_peak=max_peak)
    X_members_n2 = classifier.normalize(X_members, mode='each')

    #classifier.save_to_file(X_members_n1, y_members, 'normed_members_by_total.csv', 'normalized/members')
    classifier.save_to_file(X_members_n2, y_members, 'normalized_members_by_each.csv', 'normalized/members')

    return 0


if __name__ == '__main__':
    main()
