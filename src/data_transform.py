import numpy as np
import os

# for debugging
import matplotlib.pyplot as plt

import classifier
import features2

ROOT = classifier.ROOT
PEAK_INDEX = 17

##################################################################
# Data Transformations
##################################################################

# X = NxD matrix of N samples with D attributes

def zero_baseline(X, dt_ms=0.025):
    def per_row(row):
        argmin = np.argmin(row[:PEAK_INDEX])
        offset = np.mean(row[:argmin + 1])
        return np.array(map(lambda x: x - offset, row))

    return np.array(map(lambda row: per_row(row), X))


def normalize(X, dt_ms=0.025):
    def per_row(row):
        div = np.amax(row)
        return np.array(map(lambda x: x / div, row))

    return np.array(map(lambda row: per_row(row), X))


def bsln_norm(X, dt_ms=0.025):
    X_bsln = zero_baseline(X)
    return normalize(X_bsln)


def calc_features(X, dt_ms=0.025):
    def per_row(row):
        #peak_index = np.argmax(row)
        feats = features2.calculate_features(row, peak_index=PEAK_INDEX)
        return np.array(feats)

    return np.array(map(lambda row: per_row(row), X))


def calc_feats_naive(X, dt_ms=0.025): 
    def per_row(row):
        feats = features2.calc_feats_naive(row, dt_ms)
        return np.array(feats)
    
    return np.array(map(lambda row: per_row(row), X))


##################################################################


# convert files from srcroot using transformation func and save to destroot
def data_transform(func, srcroot, dstroot, header='label,attributes'):
    print('Executing io_transform: loading from ' + srcroot + ' to ' + dstroot)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, srcroot)):
        csvs = filter(lambda f: f.endswith('.csv'), filenames)              
        for fname in csvs:                                                  
            print('loading from file: ' + os.path.join(dirpath, fname))
            X, y = classifier.load_data(dirpath, fname)
            dt_ms = classifier.load_dt_ms(dirpath, fname)
            X_new = func(X, dt_ms=dt_ms)

            subj_sess = dirpath[(len(ROOT) + 2 + len(srcroot)) :]
            dst_dirpath = os.path.join(ROOT, dstroot, subj_sess)
            print('saving to directory: ' + dst_dirpath)
            if not os.path.exists(dst_dirpath):
                os.makedirs(dst_dirpath)

            rows = np.array(map(lambda X_i, y_i: np.append(y_i, X_i), X_new, y))

            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, np.array([dt_ms]), fmt='%g', delimiter=',', header='dt_ms')
            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, rows, fmt='%g', delimiter=',', header=header)


def get_header(src):
	fp = open(src)
	for i, line in enumerate(fp):
		if i == 2:
			return line	


def save_test(srcroot, dstroot):
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, srcroot)):
        csvs = filter(lambda f: f.endswith('.csv'), filenames)              
        for fname in csvs:                                                  
            src = os.path.join(dirpath, fname)                                  
            print('loading from file: ' + src)
            data = np.loadtxt(src, delimiter=',', skiprows=2)                 
            y = data[:, 0].astype(int)
            dt_ms = classifier.load_dt_ms(dirpath, fname)
            p_indices = np.array(map(lambda y_i: y_i == classifier.CLASS_P, y))
            i_indices = np.array(map(lambda y_i: y_i == classifier.CLASS_I, y))

            data_p = data[p_indices]
            data_i = data[i_indices]
            if data_p.shape[0] <= 0 and data_i.shape[0] <= 0:
                continue

            subj_sess = dirpath[(len(ROOT) + 2 + len(srcroot)) :]
            dst_dirpath = os.path.join(ROOT, dstroot, subj_sess)

            if not os.path.exists(dst_dirpath):
                os.makedirs(dst_dirpath)

            print('saving to directory: ' + dst_dirpath)
            if not os.path.exists(os.path.join(dst_dirpath, fname)):
                # save dt_ms
                with open(os.path.join(dst_dirpath, fname), 'w') as f:
                    np.savetxt(f, np.array([dt_ms]), fmt='%g', delimiter=',', header='dt_ms')

			# save random P and I row from file
            header = get_header(src)
            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                f.write(header)
                if data_p.shape[0] > 0 and data_i.shape[0] > 0:                                           
                    row_p = np.array([data_p[np.random.randint(0, data_p.shape[0])]])
                    row_i = np.array([data_i[np.random.randint(0, data_i.shape[0])]])
                    rows = np.append(row_p, row_i, axis=0)
                    np.savetxt(f, rows, fmt='%g', delimiter=',')
                elif data_p.shape[0] > 0:
                    row = np.array([data_p[np.random.randint(0, data_p.shape[0])]])
                    np.savetxt(f, row, fmt='%g', delimiter=',')
                elif data_i.shape[0] > 0:
					row = np.array([data_i[np.random.randint(0, data_i.shape[0])]])
					np.savetxt(f, row, fmt='%g', delimiter=',')


if __name__ == '__main__':
	save_test('raw/members', 'test/raw/members')
	save_test('bsln_norm/members', 'test/bsln_norm/members')

