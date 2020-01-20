import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dataset-path')
args = parser.parse_args()

base_path = os.path.join(os.path.abspath(args.dataset_path), 'experiment1/all_data/')
if not os.path.exists(base_path):
    raise Exception('You must convert the old "training" dir to all_data. You must do this manually')
new_training_path = os.path.join(os.path.abspath(args.dataset_path), 'experiment1/training/')
if os.path.exists(new_training_path):
    raise Exception('You must convert the old "training" dir to all_data. You must do this manually')
new_testing_path = os.path.join(os.path.abspath(args.dataset_path), 'experiment1/testing/')
if os.path.exists(new_training_path):
    raise Exception('The new testing dir should not exist. Have you run this script already?')
os.makedirs(os.path.join(new_training_path, 'raw'))
os.makedirs(os.path.join(new_training_path, 'meta'))
os.makedirs(os.path.join(new_testing_path, 'raw'))
os.makedirs(os.path.join(new_testing_path, 'meta'))

train_ards =  ['0261RPI1220160617',
 '0139RPI1620160205',
 '0149RPI1820160212',
 '0511RPI5220170831',
 '0160RPI1420160220',
 '0122RPI1320160120',
 '0235RPI1320160426',
 '0551RPI0720180102',
 '0194RPI0320160317',
 '0357RPI3520161101',
 '0271RPI1220160630',
 '0021RPI0420150513',
 '0102RPI0120151225',
 '0027RPI0620150525',
 '0266RPI1720160622',
 '0268RPI1220160624',
 '0093RPI0920151212',
 '0527RPI0420171028',
 '0260RPI2420160617',
 '0506RPI3720170807',
 '0127RPI0120160124',
 '0129RPI1620160126',
 '0640RPI2820180822',
 '0546RPI5120171216',
 '0253RPI1220160606',
 '0390RPI2220161230',
 '0412RPI5520170121',
 '0153RPI0720160217',
 '0245RPI1420160512',
 '0265RPI2920160622',
 '0209RPI1920160408',
 '0098RPI1420151218',
 '0484RPI4220170630',
 '0372RPI2220161211',
 '0251RPI1820160609']
train_other = ['0423RPI3220170205',
 '0365RPI5820161125',
 '0443RPI1620170319',
 '0387RPI3920161224',
 '0354RPI5820161029',
 '0225RPI2520160416',
 '0410RPI4120170118',
 '0124RPI1220160123',
 '0347RPI4220161016',
 '0463RPI3220170522',
 '0361RPI4620161115',
 '0111RPI1520160101',
 '0133RPI0920160127',
 '0304RPI1620160829',
 '0624RPI0320180708',
 '0317RPI3220160910',
 '0356RPI2220161101',
 '0545RPI0520171214',
 '0625RPI2820180628',
 '0231RPI1220160424',
 '0257RPI1220160615',
 '0585RPI2720180206',
 '0108RPI0120160101',
 '0460RPI2220170518',
 '0544RPI2420171204',
 '0315RPI2720160910',
 '0166RPI2220160227',
 '0434RPI4520170224',
 '0144RPI0920160212',
 '0398RPI4220170104',
 '0343RPI3920161016',
 '0624RPI1920180702',
 '0033RPI0520150603',
 '0593RPI1920180226',
 '0170RPI2120160301']
test_ards = ['0015RPI0320150401',
 '0026RPI1020150523',
 '0099RPI0120151219',
 '0120RPI1820160118',
 '0147RPI1220160213',
 '0148RPI0120160214',
 '0224RPI3020160414',
 '0243RPI0720160512',
 '0381RPI2320161212',
 '0411RPI5820170119',
 '0514RPI5420170905',
 '0549RPI4420171213',
 '0558RPI0820180104',
 '0569RPI0420180116',
 '0723RPI2120190416']
test_other = ['0112RPI1620160105',
 '0125RPI1120160123',
 '0132RPI1720160127',
 '0135RPI1420160203',
 '0137RPI1920160202',
 '0145RPI1120160212',
 '0157RPI0920160218',
 '0163RPI0720160222',
 '0173RPI1920160303',
 '0306RPI3520160830',
 '0336RPI3920161006',
 '0380RPI3920161212',
 '0552RPI2520180101',
 '0705RPI5020190318',
 '0745RPI1900000000']

for pt in train_ards + train_other:
    proc = subprocess.Popen(['ln', '-s', base_path + 'raw/' + pt, new_training_path + 'raw/'])
    proc.communicate()
    proc = subprocess.Popen(['ln', '-s', base_path + 'meta/' + pt, new_training_path + 'meta/'])
    proc.communicate()

for pt in test_ards + test_other:
    proc = subprocess.Popen(['ln', '-s', base_path + 'raw/' + pt, new_testing_path + 'raw/'])
    proc.communicate()
    proc = subprocess.Popen(['ln', '-s', base_path + 'meta/' + pt, new_testing_path + 'meta/'])
    proc.communicate()
