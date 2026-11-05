import torch
import sys
import numpy as np
import os

try:
    # Optional dependency; only needed for specific synthetic generators
    from txai.synth_data.simple_spike import SpikeTrainDataset
except Exception:
    SpikeTrainDataset = None
from txai.utils.data.preprocess import process_MITECG, process_Boiler
from txai.utils.data import process_Synth
from txai.utils.data.synth import process_Synth_named
from colax import interpolate_tensor

DATA_DICT = {   
    "adiac": "Adiac",
    "arrowhead": "ArrowHead",
    "beef": "Beef",
    "beetlefly": "BeetleFly",
    "birdchicken": "BirdChicken",
    "car": "Car",
    "cbf": "CBF",
    "chlorineconcentration": "ChlorineConcentration",
    "cincecgtorso": "CinCECGTorso",
    "coffee": "Coffee",
    "computers": "Computers",
    "cricketx": "CricketX",
    "crickety": "CricketY",
    "cricketz": "CricketZ",
    "diatomsizereduction": "DiatomSizeReduction",
    "distalphalanxoutlineagegroup": "DistalPhalanxOutlineAgeGroup",
    "distalphalanxoutlinecorrect": "DistalPhalanxOutlineCorrect",
    "distalphalanxtw": "DistalPhalanxTW",
    "earthquakes": "Earthquakes",
    "ecg200": "ECG200",
    "ecg5000": "ECG5000",
    "ecgfivedays": "ECGFiveDays",
    "electricdevices": "ElectricDevices",
    "faceall": "FaceAll",
    "facefour": "FaceFour",
    "facesucr": "FacesUCR",
    "fiftywords": "FiftyWords",
    "fish": "Fish",
    "forda": "FordA",
    "fordb": "FordB",
    "gunpoint": "GunPoint",
    "ham": "Ham",
    "handoutlines": "HandOutlines",
    "haptics": "Haptics",
    "herring": "Herring",
    "inlineskate": "InlineSkate",
    "insectwingbeatsound": "InsectWingbeatSound",
    "italypowerdemand": "ItalyPowerDemand",
    "largekitchenappliances": "LargeKitchenAppliances",
    "lightning2": "Lightning2",
    "lightning7": "Lightning7",
    "mallat": "Mallat",
    "meat": "Meat",
    "medicalimages": "MedicalImages",
    "middlephalanxoutlineagegroup": "MiddlePhalanxOutlineAgeGroup",
    "middlephalanxoutlinecorrect": "MiddlePhalanxOutlineCorrect",
    "middlephalanxtw": "MiddlePhalanxTW",
    "motestrain": "MoteStrain",
    "noninvasivefetalecgthorax1": "NonInvasiveFetalECGThorax1",
    "noninvasivefetalecgthorax2": "NonInvasiveFetalECGThorax2",
    "oliveoil": "OliveOil",
    "osuleaf": "OSULeaf",
    "phalangesoutlinescorrect": "PhalangesOutlinesCorrect",
    "phoneme": "Phoneme",
    "plane": "Plane",
    "proximalphalanxoutlineagegroup": "ProximalPhalanxOutlineAgeGroup",
    "proximalphalanxoutlinecorrect": "ProximalPhalanxOutlineCorrect",
    "proximalphalanxtw": "ProximalPhalanxTW",
    "refrigerationdevices": "RefrigerationDevices",
    "screentype": "ScreenType",
    "shapeletsim": "ShapeletSim",
    "shapesall": "ShapesAll",
    "smallkitchenappliances": "SmallKitchenAppliances",
    "sonyaiborobotsurface1": "SonyAIBORobotSurface1",
    "sonyaiborobotsurface2": "SonyAIBORobotSurface2",
    "starlightcurves": "StarLightCurves",
    "strawberry": "Strawberry",
    "swedishleaf": "SwedishLeaf",
    "symbols": "Symbols",
    "syntheticcontrol": "SyntheticControl",
    "toesegmentation1": "ToeSegmentation1",
    "toesegmentation2": "ToeSegmentation2",
    "trace": "Trace",
    "twoleadecg": "TwoLeadECG",
    "twopatterns": "TwoPatterns",
    "uwavegesturelibraryall": "UWaveGestureLibraryAll",
    "uwavegesturelibraryx": "UWaveGestureLibraryX",
    "uwavegesturelibraryy": "UWaveGestureLibraryY",
    "uwavegesturelibraryz": "UWaveGestureLibraryZ",
    "wafer": "Wafer",
    "wine": "Wine",
    "wordsynonyms": "WordSynonyms",
    "worms": "Worms",
    "wormstwoclass": "WormsTwoClass",
    "yoga": "Yoga",
    "acsf1": "ACSF1",
    "allgesturewiimotex": "AllGestureWiimoteX",
    "allgesturewiimotey": "AllGestureWiimoteY",
    "allgesturewiimotez": "AllGestureWiimoteZ",
    "bme": "BME",
    "chinatown": "Chinatown",
    "crop": "Crop",
    "dodgerloopday": "DodgerLoopDay",
    "dodgerloopgame": "DodgerLoopGame",
    "dodgerloopweekend": "DodgerLoopWeekend",
    "eoghorizontalsignal": "EOGHorizontalSignal",
    "eogverticalsignal": "EOGVerticalSignal",
    "ethanollevel": "EthanolLevel",
    "freezerregulartrain": "FreezerRegularTrain",
    "freezersmalltrain": "FreezerSmallTrain",
    "fungi": "Fungi",
    "gesturemidaird1": "GestureMidAirD1",
    "gesturemidaird2": "GestureMidAirD2",
    "gesturemidaird3": "GestureMidAirD3",
    "gesturepebblez1": "GesturePebbleZ1",
    "gesturepebblez2": "GesturePebbleZ2",
    "gunpointagespan": "GunPointAgeSpan",
    "gunpointmaleversusfemale": "GunPointMaleVersusFemale",
    "gunpointoldversusyoung": "GunPointOldVersusYoung",
    "housetwenty": "HouseTwenty",
    "insectepgregulartrain": "InsectEPGRegularTrain",
    "insectepgsmalltrain": "InsectEPGSmallTrain",
    "melbournepedestrian": "MelbournePedestrian",
    "mixedshapesregulartrain": "MixedShapesRegularTrain",
    "mixedshapessmalltrain": "MixedShapesSmallTrain",
    "pickupgesturewiimotez": "PickupGestureWiimoteZ",
    "pigairwaypressure": "PigAirwayPressure",
    "pigartpressure": "PigArtPressure",
    "pigcvp": "PigCVP",
    "plaid": "PLAID",
    "powercons": "PowerCons",
    "rock": "Rock",
    "semghandgenderch2": "SemgHandGenderCh2",
    "semghandmovementch2": "SemgHandMovementCh2",
    "semghandsubjectch2": "SemgHandSubjectCh2",
    "shakegesturewiimotez": "ShakeGestureWiimoteZ",
    "smoothsubspace": "SmoothSubspace",
    "umd": "UMD",
    # ---- UEA Multivariate Datasets ----
    "epilepsy": "Epilepsy",
    "atrialfibrillation": "AtrialFibrillation",
    "selfregulationscp1": "SelfRegulationSCP1",
    "heartbeat": "Heartbeat",
}



# ---- UEA dataset dimensions: (n_channels, seq_len) ----
UEA_DIMS = {
    "Epilepsy":              (3, 206),
    "AtrialFibrillation":    (2, 640),
    "SelfRegulationSCP1":    (6, 896),
    "Heartbeat":             (61, 405),
}


def _append_timex_root_if_any(args):
    # Timex path no longer configured; keep placeholder for backward compatibility
    return


class SynthTrainDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X, self.times, self.y = X, times, y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 
    
class SynthTrainDataset_Batch_first(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X, self.times, self.y = X, times, y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx,:,:]
        T = self.times[idx,:]
        y = self.y[idx]
        return x, T, y 
    
    
class Get_torch_dataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, T_first=False,interpolate=False):
        
        print(f"[Get_torch_dataset.__init__] Input X shape: {X.shape}, T_first={T_first}")
        
        if T_first: # ECG dataset
            X = X.transpose(1,0) # Shape: (T, N, d) ->(N, T, d)
            print(f"[Get_torch_dataset.__init__] After transpose: {X.shape}")
        
        self.X = X
        self.y = y # Shape: (N,)
        
        if interpolate:
            self.X = interpolate_tensor(self.X,new_T=500,dim_to_change=1)
        
        # print(f"[Get_torch_dataset.__init__] Final X shape: {self.X.shape}, X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
        # print(f"[Get_torch_dataset.__init__] Y shape: {self.y.shape}, unique: {self.y.unique()}")
        
        self.max_seq_len=self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx,:,:]
        #T = self.times[:,idx]
        y = self.y[idx]
        

        return x, y , torch.ones((x.shape[0],x.shape[1],128))

class Get_ECG_dataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, augment_negative = None):
        self.X = X.transpose(1,0) # Shape: (T, N, d) ->(N, T, d) 
        #self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)
        # print('X', self.X.shape)
        # print('times', self.times.shape)
        # print('y', self.y.shape)
        # exit()
        self.max_seq_len=self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx,one_hot=False):
        x = self.X[idx,:,:]
        #T = self.times[:,idx]
        y = self.y[idx]
        device = x.device
        
        if one_hot:
            ONE_HOT=torch.tensor([[1,0],[0,1]]).to(device)
            y_one_hot=ONE_HOT[y]
        return x, y , torch.ones((x.shape[0],x.shape[1],128))





# ---> Load UEA multivariate data

def get_UEA_data(dataset_name, device, Swap_testTrain=True, shuffle_train=True,
                 if_normalize=False, return_dict=True, root_path=None):
    """Load a UEA multivariate time-series dataset.

    UEA TSV format: first column = label, remaining = D×T features
    (channel-major: all timesteps of ch0, then all timesteps of ch1, ...).

    Returns
    -------
    If return_dict: train_dataset, val_dataset, test_dataset, (D, T)
    If not: train, val, test tuples (X, T, Y) where X is (T, N, D), plus (D, T)
    """
    if root_path is None:
        root_path = "."

    possible_paths = [
        os.path.join(root_path, "datasets", "UEA"),
        os.path.join(root_path, "Colax-main", "datasets", "UEA"),
    ]

    input_path = None
    for path in possible_paths:
        if os.path.exists(path):
            input_path = path
            break

    if input_path is None:
        raise FileNotFoundError(
            f"UEA dataset directory not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths))

    ds_display = DATA_DICT.get(dataset_name.lower(), dataset_name)
    D, T_dim = UEA_DIMS.get(ds_display, (None, None))

    print(f"Loading UEA data: {ds_display}".center(80, "-"))
    print(f"  (D={D}, T={T_dim})" if D else "  (dimensions unknown)")

    train_path = os.path.join(input_path, ds_display, f"{ds_display}_TRAIN.tsv")
    test_path = os.path.join(input_path, ds_display, f"{ds_display}_TEST.tsv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    training_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    # Infer dimensions from data if not known
    if D is None or T_dim is None:
        n_cols = training_data.shape[1] - 1  # exclude label column
        # Try to infer: we need D and T such that D * T = n_cols
        # For now, assume single channel (T = n_cols, D = 1)
        D = 1
        T_dim = n_cols
        print(f"  Inferred: D={D}, T={T_dim}")

    if Swap_testTrain:
        training_data, test_data = test_data, training_data

    if shuffle_train:
        np.random.shuffle(training_data)

    val_to_test_ratio = 0.2
    num_val = int(training_data.shape[0] * val_to_test_ratio)

    val_data = training_data[:num_val]
    training_data = training_data[num_val:]

    Y_test = test_data[:, 0].astype(np.int32)
    X_test = test_data[:, 1:].reshape(-1, D, T_dim).transpose(0, 2, 1)  # (N, T, D)

    Y_training = training_data[:, 0].astype(np.int32)
    X_training = training_data[:, 1:].reshape(-1, D, T_dim).transpose(0, 2, 1)

    Y_val = val_data[:, 0].astype(np.int32)
    X_val = val_data[:, 1:].reshape(-1, D, T_dim).transpose(0, 2, 1)

    # Remap labels to 0..C-1
    all_labels = np.unique(np.concatenate([Y_training, Y_val, Y_test]))
    label_map = {lab: idx for idx, lab in enumerate(sorted(all_labels))}
    print(f"  Labels: {label_map}")
    Y_training = np.array([label_map[y] for y in Y_training], dtype=np.int64)
    Y_val = np.array([label_map[y] for y in Y_val], dtype=np.int64)
    Y_test = np.array([label_map[y] for y in Y_test], dtype=np.int64)

    n_classes = len(all_labels)
    print(f"  Classes: {n_classes}, "
          f"Train: {X_training.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    if if_normalize:
        mu = X_training.mean()
        std = X_training.std() + 1e-8
        X_training = (X_training - mu) / std
        X_val = (X_val - mu) / std
        X_test = (X_test - mu) / std

    # Build tensors — (T, N, D) format (time-first, matching UCR convention)
    X_train_t = torch.tensor(X_training, dtype=torch.float32).permute(1, 0, 2)  # (T, N, D)
    times_train = torch.tensor(np.arange(T_dim), dtype=torch.float32).unsqueeze(1).repeat(1, X_train_t.shape[1])
    Y_train_t = torch.tensor(Y_training, dtype=torch.int64)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device).permute(1, 0, 2)
    times_val = torch.tensor(np.arange(T_dim), dtype=torch.float32).to(device).unsqueeze(1).repeat(1, X_val_t.shape[1])
    Y_val_t = torch.tensor(Y_val, dtype=torch.int64).to(device)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device).permute(1, 0, 2)
    times_test = torch.tensor(np.arange(T_dim), dtype=torch.float32).to(device).unsqueeze(1).repeat(1, X_test_t.shape[1])
    Y_test_t = torch.tensor(Y_test, dtype=torch.int64).to(device)

    train = [X_train_t, times_train, Y_train_t]
    val = [X_val_t, times_val, Y_val_t]
    test = [X_test_t, times_test, Y_test_t]

    if return_dict:
        train_dataset = Get_torch_dataset(train[0], train[1], train[2], T_first=True)
        val_dataset = Get_torch_dataset(val[0], val[1], val[2], T_first=True)
        test_dataset = Get_torch_dataset(test[0], test[1], test[2], T_first=True)
        return train_dataset, val_dataset, test_dataset, (D, T_dim)
    else:
        return train, val, test, (D, T_dim)


def get_saliency_data(args,return_dict=False,split_no=301):
    """
    
    return: train,val,test: 
    train(X,T,Y)
    val(X,T,Y)
    test(X,T,Y)
    """
    
    dataset_name =args.data
    device = args.device
    meta_dataset = getattr(args, 'meta_dataset', 'default')
    
    # Auto-detect UCR datasets: if dataset name is in DATA_DICT and meta_dataset is 'default',
    # automatically set meta_dataset to 'ucr'
    if meta_dataset == 'default' and dataset_name.lower() in DATA_DICT:
        print(f"Auto-detected UCR dataset: {dataset_name} -> {DATA_DICT[dataset_name.lower()]}")
        print(f"Setting meta_dataset to 'ucr'")
        meta_dataset = 'ucr'
        args.meta_dataset = 'ucr'  # Update args for consistency
    
    # Optionally extend sys.path for timex utils if provided
    _append_timex_root_if_any(args)
    
    if meta_dataset == 'ucr':
        pass
    
    elif args.data.rsplit("_", 1)[0] == "freqshape":
        pass
    
    elif dataset_name in ["mcce","mcch","mtce","mtch"]:
        # treat as freqshape splits with new names: mcce->301, mcch->303, mtce->311, mtch->315
        pass
    
    elif dataset_name == 'mitecg':
        DATASET_PATH = None
    
    else:
        # fallback only for known dictionary datasets
        DATASET_PATH = None
    
    # ---- custom pure-tensor dataset: tipping ----
    if 'tipping' in dataset_name:
        base_path = os.path.join(args.root_path or '.', 'datasets')
        tipping_path = os.path.join(base_path, f'{dataset_name}.pt')
        if not os.path.exists(tipping_path):
            raise FileNotFoundError(f"tipping.pt not found at {tipping_path}")

        D = torch.load(tipping_path, map_location=device)

        def _fix_tuple(t):
            X, Tm, Y = t
            X = X.float().to(device)
            Tm = Tm.float().to(device)
            Y = Y.long().to(device)
            # ensure X has channel dim (T,N,d)
            if X.ndim == 2:
                X = X.unsqueeze(-1)
            return X, Tm, Y

        train = _fix_tuple(D['train'])
        val = _fix_tuple(D['val'])
        test = _fix_tuple(D['test'])
        gt_exps = D.get('gt_exps', None)
        if gt_exps is not None:
            gt_exps = gt_exps.to(device).float()

        # Wrap for training/eval pipeline (non-saliency)
        train_dataset = Get_torch_dataset(train[0], train[1], train[2], T_first=True)
        val_dataset = Get_torch_dataset(val[0], val[1], val[2], T_first=True)
        test_dataset = Get_torch_dataset(test[0], test[1], test[2], T_first=True)

        data_dict = {
            "TEST": test_dataset,
            "VAL": val_dataset,
            "TRAIN": train_dataset,
        }

        if args.saliency:
            # Return batch-first test tensors and gt_exps aligned as (N,T,1)
            test_X = test[0].transpose(0, 1)
            test_T = test[1].transpose(0, 1)
            test_Y = test[2]
            if gt_exps is None:
                gt = torch.zeros((test_X.shape[0], test_X.shape[1], 1), device=test_X.device, dtype=test_X.dtype)
            else:
                # assume stored as (T,N,1)
                gt = gt_exps.transpose(0, 1)
            return test_X, test_Y, test_T, gt
        else:
            return data_dict

    # ---- AMOC Early-Warning Dataset ----
    if 'amoc' in dataset_name.lower():
        base_path = os.path.join(args.root_path or '.', 'datasets')
        # Support both 'amoc' and 'amoc_xxx' naming
        amoc_path = os.path.join(base_path, f'{dataset_name}.pt')
        if not os.path.exists(amoc_path):
            # Try default name
            amoc_path = os.path.join(base_path, 'amoc.pt')
        if not os.path.exists(amoc_path):
            raise FileNotFoundError(
                f"AMOC dataset not found. Expected one of:\n"
                f"  - {os.path.join(base_path, f'{dataset_name}.pt')}\n"
                f"  - {amoc_path}\n"
                f"Please run: python download_amoc.py and python process_amoc.py"
            )

        D = torch.load(amoc_path, map_location=device)

        def _fix_tuple(t):
            X, Tm, Y = t
            X = X.float().to(device)
            Tm = Tm.float().to(device)
            Y = Y.long().to(device)
            # ensure X has channel dim (T,N,d)
            if X.ndim == 2:
                X = X.unsqueeze(-1)
            return X, Tm, Y

        train = _fix_tuple(D['train'])
        val = _fix_tuple(D['val'])
        test = _fix_tuple(D['test'])
        gt_exps = D.get('gt_exps', None)
        if gt_exps is not None:
            gt_exps = gt_exps.to(device).float()

        # Wrap for training/eval pipeline (non-saliency)
        train_dataset = Get_torch_dataset(train[0], train[1], train[2], T_first=True)
        val_dataset = Get_torch_dataset(val[0], val[1], val[2], T_first=True)
        test_dataset = Get_torch_dataset(test[0], test[1], test[2], T_first=True)

        data_dict = {
            "TEST": test_dataset,
            "VAL": val_dataset,
            "TRAIN": train_dataset,
        }

        if args.saliency:
            # Return batch-first test tensors and gt_exps aligned as (N,T,1)
            test_X = test[0].transpose(0, 1)
            test_T = test[1].transpose(0, 1)
            test_Y = test[2]
            if gt_exps is None:
                gt = torch.zeros((test_X.shape[0], test_X.shape[1], 1), device=test_X.device, dtype=test_X.dtype)
            else:
                # assume stored as (T,N,1)
                gt = gt_exps.transpose(0, 1)
            return test_X, test_Y, test_T, gt
        else:
            return data_dict

    if args.data.rsplit("_", 1)[0] == "freqshape":
        split_no = int(args.data.rsplit("_", 1)[1])
        
        print("TTTTTTTTTTTest:")
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth(split_no = split_no, device = device, base_path = base_path)
        
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)

        val, test,  = D['val'], D['test']
        
        print(f"Debug - val[0].shape: {val[0].shape}, val[1].shape: {val[1].shape}, val[2].shape: {val[2].shape}")
        print(f"Debug - test[0].shape: {test[0].shape}, test[1].shape: {test[1].shape}, test[2].shape: {test[2].shape}")
        
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0])
            T.append(item[1])
            Y.append(item[2])
            
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1).long()
        train=(X,T,Y)
        
        test_y=test[2]
        #gt_exps=D['gt_exps'][:,(test_y != 0).detach().cpu(),:]
        gt_exps=D['gt_exps'][:,:,:]
        
        ##### get training data
        
        
        train_dataset=Get_torch_dataset(train[0],train[1],train[2]) #!!!
        val_dataset=Get_torch_dataset(val[0][:,:,:],val[1][:,:],val[2][:],T_first=True,interpolate=False)

        #val_dataset=Get_torch_dataset(val[0],val[1],val[2],None,True)
        #val_dataset=Get_torch_dataset(test[0][:,950:1050,:],test[1][:,950:1050],test[2][950:1050],None,True) # freqshape 274
        test_dataset=Get_torch_dataset(test[0][:,:,:],test[1][:,:],test[2][:],T_first=True,interpolate=False)


        data_dict={ # 
            "TEST":train_dataset,
            "VAL":val_dataset,
            "TRAIN":test_dataset
        }
        print(X.shape)
        
        #####
        test_X=test[0]
        test_T=test[1]
        test_Y=test[2]
        test_X= test_X.transpose(0,1)
        test_T= test_T.transpose(0,1)
        
        if args.saliency:
            gt_exps = gt_exps.transpose(0,1)  # (N,T,1)
            return  test_X ,test_Y,test_T, gt_exps
        else:
            return data_dict
    
    elif dataset_name in ['mcce','mcch','mtce','mtch']:
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth_named(dataset_name, device=device, base_path=base_path)
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)
        val, test,  = D['val'], D['test']
        print(f"Debug - val[0].shape: {val[0].shape}, val[1].shape: {val[1].shape}, val[2].shape: {val[2].shape}")
        print(f"Debug - test[0].shape: {test[0].shape}, test[1].shape: {test[1].shape}, test[2].shape: {test[2].shape}")
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0]); T.append(item[1]); Y.append(item[2])
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1).long()
        train=(X,T,Y)
        gt_exps=D['gt_exps'][:,:,:]
        train_dataset=Get_torch_dataset(train[0],train[1],train[2])
        val_dataset=Get_torch_dataset(val[0][:,:,:],val[1][:,:],val[2][:],T_first=True,interpolate=False)
        test_dataset=Get_torch_dataset(test[0][:,:,:],test[1][:,:],test[2][:],T_first=True,interpolate=False)
        data_dict={"TEST":train_dataset,"VAL":val_dataset,"TRAIN":test_dataset}
        print(X.shape)
        test_X=test[0]; test_T=test[1]; test_Y=test[2]
        test_X= test_X.transpose(0,1)
        test_T= test_T.transpose(0,1)
        if args.saliency:
            gt_exps = gt_exps.transpose(0,1)  # (N,T,1)
            return  test_X ,test_Y,test_T, gt_exps
        else:
            return data_dict
    
    # removed the separate local pt-loading branch for mcce/mcch/mtce/mtch to keep logic identical to freqshape
    
    elif dataset_name == 'mitecg':

        #D = process_Synth(split_no = 1, device = device, base_path = DATASET_PATH)
        base_path = os.path.join(args.root_path or '.', 'datasets', 'mitecg')
        D = process_MITECG(split_no = 1, device = device, hard_split = True, need_binarize = True, exclude_pac_pvc = True, base_path = base_path)

        train, val, test, gt_exps = D

        X, times, y = test.X, test.time, test.y 

        print(X.shape)




        mask = (y == 1).clone().cpu() ##### !!! orignal setting
        
        
        if args.is_training:
            print("don’t use mitecg one class mask",)
            mask = (y == 1) | (y == 0)

        
        mask=mask.clone().cpu()
        print("###### main: mask.shape: ",mask.shape)

        # Detect when we fail to observe a wave:
        detection_failure = (gt_exps.squeeze().sum(0) > 0) 
        mask = mask & detection_failure ##### !!! orignal setting
        X = X[:,mask,:]
        times = times[:,mask]
        gt_exps = gt_exps[:,mask,:]
        y = y[mask]

        # convert to batch-first for saliency returns
        X = X.transpose(0,1)
        times = times.transpose(0,1)
        gt_exps = gt_exps.transpose(0,1)


        #output,attens=model.classification(x,x_mark_enc)

        # for  ECG dataset
        train_dataset=Get_ECG_dataset(train.X,train.time,train.y)
        val_dataset=Get_ECG_dataset(val.X,val.time,val.y)
        test_dataset=Get_ECG_dataset(X,times,y)

        data_dict={ # 
            "TEST":test_dataset,
            "VAL":val_dataset,
            "TRAIN":train_dataset
        }

        print(X.shape)


        if args.saliency:
            return  X ,y,times, gt_exps
        else:
            return data_dict

    
    
    elif dataset_name == 'seqcomb_single':
        print("TTTTTTTTTTTest:")
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth(split_no = split_no, device = device, base_path =base_path)
        
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)

        val, test,  = D['val'], D['test']
        
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0])
            T.append(item[1])
            Y.append(item[2])
            
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1)
        train=(X,T,Y)
        
        test_y=test[2]
        #gt_exps=D['gt_exps'][:,(test_y != 0).detach().cpu(),:]
        gt_exps=D['gt_exps'][:,:,:]
        
    elif dataset_name == 'seqcomb_single_better':
        print("TTTTTTTTTTTest:")
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth(split_no = split_no, device = device, base_path =base_path)
        
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)

        val, test,  = D['val'], D['test']
        
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0])
            T.append(item[1])
            Y.append(item[2])
            
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1)
        train=(X,T,Y)
        
        test_y=test[2]
        #gt_exps=D['gt_exps'][:,(test_y != 0).detach().cpu(),:]
        gt_exps=D['gt_exps'][:,:,:]
        
        
    elif meta_dataset == 'ucr':
        root_path = getattr(args, 'root_path', None)
        
        # For saliency mode, we need raw tensors, not datasets
        if args.saliency:
            # Get raw data tuples (X, T, Y) format
            train_tuple, val_tuple, test_tuple, gt_exps = get_URC_data(
                dataset_name, device,
                Swap_testTrain=True,
                shuffle_train=True,
                return_train_loader=False,
                train_seq_first=False,
                if_normalize=False,
                test_concat_val=False,
                return_dict=False,  # Get raw tuples for saliency mode
                root_path=root_path
            )
            
            # Convert to batch-first format for saliency: (N, T, 1)
            # test_tuple is (X, T, Y) where X is (T, N, 1), T is (T, N), Y is (N,)
            test_X = test_tuple[0].transpose(0, 1)  # (T, N, 1) -> (N, T, 1)
            test_T = test_tuple[1].transpose(0, 1)  # (T, N) -> (N, T)
            test_Y = test_tuple[2]  # (N,)
            
            # UCR datasets don't have ground truth explanations, create zeros
            if gt_exps is None:
                gt_exps = torch.zeros((test_X.shape[0], test_X.shape[1], 1), 
                                     device=test_X.device, dtype=test_X.dtype)
            else:
                # If gt_exps exists, ensure it's in (N, T, 1) format
                if gt_exps.dim() == 3 and gt_exps.shape[0] != test_X.shape[0]:
                    gt_exps = gt_exps.transpose(0, 1)
            
            return test_X, test_Y, test_T, gt_exps
        else:

            if_normalize = getattr(args, 'normalize_data', False)
            train, val, test, gt_exps = get_URC_data(
                dataset_name, device,
                Swap_testTrain=True,
                shuffle_train=True,
                return_train_loader=False,
                train_seq_first=False,
                if_normalize=if_normalize,
                test_concat_val=False,
                return_dict=return_dict,
                root_path=root_path
            )
            if return_dict:
                # For UCR datasets, we need to support both:
                # 1. List access: data_dict[0]=train, data_dict[1]=val, data_dict[2]=test (for exp_saliency_general_hybrid)
                # 2. Dict access: data_dict['TRAIN'], data_dict['VAL'], data_dict['TEST'] (for visualize_prototypes_separated)
                # Create a class that supports both access patterns
                class UCRDataDict:
                    def __init__(self, train, val, test):
                        self._data = [train, val, test]
                        self._dict = {"TRAIN": train, "VAL": val, "TEST": test}
                    
                    def __getitem__(self, key):
                        if isinstance(key, (int, slice)):
                            return self._data[key]
                        else:
                            return self._dict[key]
                    
                    def get(self, key, default=None):
                        return self._dict.get(key, default)
                    
                    def __len__(self):
                        return len(self._data)
                
                return UCRDataDict(train, val, test)
            else:
                return train, val, test, gt_exps
    
    elif meta_dataset == 'uea':
        root_path = getattr(args, 'root_path', None)

        if args.saliency:
            train_tuple, val_tuple, test_tuple, (D, T_dim) = get_UEA_data(
                dataset_name, device,
                Swap_testTrain=True,
                shuffle_train=True,
                if_normalize=False,
                return_dict=False,
                root_path=root_path,
            )
            # Store dimensions on args for downstream use
            args.n_variables = D
            args.seq_len = T_dim
            # Convert to batch-first: (T, N, D) → (N, T, D)
            test_X = test_tuple[0].transpose(0, 1)
            test_T = test_tuple[1].transpose(0, 1)
            test_Y = test_tuple[2]
            # No ground truth for UEA; use zero tensor with shape (N, T, D)
            gt_exps = torch.zeros((test_X.shape[0], test_X.shape[1], D),
                                  device=test_X.device, dtype=test_X.dtype)
            return test_X, test_Y, test_T, gt_exps
        else:
            train, val, test, (D, T_dim) = get_UEA_data(
                dataset_name, device,
                Swap_testTrain=True,
                shuffle_train=True,
                if_normalize=getattr(args, 'normalize_data', False),
                return_dict=return_dict,
                root_path=root_path,
            )
            if return_dict:
                class UEADataDict:
                    def __init__(self, tr, va, te):
                        self._data = [tr, va, te]
                        self._dict = {"TRAIN": tr, "VAL": va, "TEST": te}
                    def __getitem__(self, key):
                        if isinstance(key, (int, slice)):
                            return self._data[key]
                        return self._dict[key]
                    def get(self, key, default=None):
                        return self._dict.get(key, default)
                    def __len__(self):
                        return len(self._data)
                return UEADataDict(train, val, test)
            else:
                return train, val, test, (D, T_dim)

    # If no matching dataset found, return None or raise error
    raise ValueError(f"Dataset '{dataset_name}' with meta_dataset '{meta_dataset}' not supported or not found.")


# ---> Load UCR data
def get_URC_data(dataset_name,device,Swap_testTrain=True,shuffle_train=True,return_train_loader=False,train_seq_first=False,if_normalize=False,test_concat_val=False,return_dict=True,root_path=None):
    # Clear any previous CUDA errors before starting
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'\n!!! Stale CUDA error detected !!!')
            print(f'    This error originated from a PREVIOUS dataset training step.')
            print(f'    Rerun with: CUDA_LAUNCH_BLOCKING=1 python ...')
            print(f'    Error: {e}')
        torch.cuda.empty_cache()
    
    # Support multiple possible paths for UCR dataset
    if root_path is None:
        root_path = "."
    
    # Try multiple possible locations
    possible_paths = [
        os.path.join(root_path, "datasets", "UCR"),
        os.path.join(root_path, "UCRArchive_2018"),
        os.path.join(root_path, "Colax-main", "datasets", "UCR"),
        os.path.join(root_path, "Colax-main", "UCRArchive_2018"),
        # "/home/hbs/TS/XTS/Colax/datasets/UCR",  # Fallback to original hardcoded path
    ]
    
    input_path = None
    for path in possible_paths:
        if os.path.exists(path):
            input_path = path
            break
    
    if input_path is None:
        raise FileNotFoundError(
            f"UCR dataset directory not found. Tried the following paths:\n" +
            "\n".join([f"  - {p}" for p in possible_paths]) +
            f"\n\nPlease ensure your UCR dataset is located in one of these directories."
        )
    
    dataset_name=DATA_DICT[dataset_name]

    print(f"Loading data: {dataset_name}".center(80, "-"))

    # -- read data -------------------------------------------------------------

    print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)

    training_data = np.loadtxt(f"{input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv")
    test_data = np.loadtxt(f"{input_path}/{dataset_name}/{dataset_name}_TEST.tsv")
    
    if Swap_testTrain:
        training_data,test_data=test_data,training_data
        
    if shuffle_train:
        np.random.shuffle(training_data)
        
    val_to_test_ratio=0.2    #
    num_val = int(training_data.shape[0] * (val_to_test_ratio))
    
    val_data = training_data[:num_val]
    training_data = training_data[num_val:]
    
    if test_concat_val:
        test_data=np.concatenate((test_data,val_data),axis=0)

    Y_test, X_test, = test_data[:, 0].astype(np.int32), test_data[:, 1:]
    Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    Y_val, X_val = val_data[:, 0].astype(np.int32), val_data[:, 1:]
    
    if if_normalize:

        mu = X_training.mean()
        std = X_training.std()

        X_training = (X_training - mu) / (std + 1e-8)
        X_val = (X_val - mu) / (std + 1e-8)
        X_test = (X_test - mu) / (std + 1e-8)

 
    val = []
    train=[] 
    test=[] 
    
    # Inspect label ranges of each split
    print("\nAnalyzing label ranges:")
    print(f"Training labels - min: {Y_training.min()}, max: {Y_training.max()}, unique: {np.unique(Y_training)}")
    print(f"Validation labels - min: {Y_val.min()}, max: {Y_val.max()}, unique: {np.unique(Y_val)}")
    print(f"Test labels - min: {Y_test.min()}, max: {Y_test.max()}, unique: {np.unique(Y_test)}")
    
    # Decide whether labels need remapping
    all_labels = np.unique(np.concatenate([Y_training, Y_val, Y_test]))
    min_label = all_labels.min()
    max_label = all_labels.max()
    num_classes = len(all_labels)
    
    print(f"\nDataset statistics:")
    print(f"Number of unique classes: {num_classes}")
    print(f"Label range: [{min_label}, {max_label}]")
    
    # Always compact labels to [0, num_classes-1], regardless of gaps
    label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
    print(f"Label mapping: {label_map}")
    Y_training = np.array([label_map[y] for y in Y_training], dtype=np.int64)
    Y_val = np.array([label_map[y] for y in Y_val], dtype=np.int64)
    Y_test = np.array([label_map[y] for y in Y_test], dtype=np.int64)
    
    # Validate labels before moving to CUDA
    assert Y_training.min() >= 0 and Y_training.max() < num_classes, \
        f"Training labels out of range: min={Y_training.min()}, max={Y_training.max()}, num_classes={num_classes}"
    assert Y_val.min() >= 0 and Y_val.max() < num_classes, \
        f"Validation labels out of range: min={Y_val.min()}, max={Y_val.max()}, num_classes={num_classes}"
    assert Y_test.min() >= 0 and Y_test.max() < num_classes, \
        f"Test labels out of range: min={Y_test.min()}, max={Y_test.max()}, num_classes={num_classes}"
    
    # Normalize data if requested
    if if_normalize:

        # Calculate mean and std from training data
        X_training_np = np.array(X_training)
        mu = X_training_np.mean()
        std = X_training_np.std()
        
        # Normalize all splits using training statistics
        X_training = (X_training_np - mu) / (std + 1e-8)
        X_val = (np.array(X_val) - mu) / (std + 1e-8)
        X_test = (np.array(X_test) - mu) / (std + 1e-8)

    
    # Ensure data is numpy array before tensor conversion
    if not isinstance(X_training, np.ndarray):
        X_training = np.array(X_training)
    if not isinstance(X_val, np.ndarray):
        X_val = np.array(X_val)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    
    # Validate data before tensor conversion
    # Check for NaN and Inf
    if np.isnan(X_training).any():
        print("Warning: X_training contains NaN values, replacing with 0")
        X_training = np.nan_to_num(X_training, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(X_val).any():
        print("Warning: X_val contains NaN values, replacing with 0")
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(X_test).any():
        print("Warning: X_test contains NaN values, replacing with 0")
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check data shapes
    print(f"Data shapes - X_training: {X_training.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    
    # train - Create tensor on CPU first, then move to device
    try:
        X_training = torch.tensor(X_training, dtype=torch.float32)

        X_training = X_training.unsqueeze(-1)
    except RuntimeError as e:
        error_msg = str(e)
        print(f"Error creating X_training tensor: {error_msg}")
        print(f"X_training shape: {X_training.shape if hasattr(X_training, 'shape') else 'N/A'}, dtype: {type(X_training)}")
        

    # Always convert to (T, N, d) format to match val and test
    if not train_seq_first:
        X_training=X_training.permute(1,0,2)  # (N, T, d) -> (T, N, d)
    
    # times_training=torch.tensor(np.arange(X_training.shape[0])).float().to(device)
    times_training = torch.tensor(np.arange(X_training.shape[0]))
    times_training=times_training.unsqueeze(1).repeat(1,X_training.shape[1])
    
    train.append(X_training)
    train.append(times_training)
    train.append(Y_training)
    
    # val 
    X_val=torch.tensor(X_val, dtype=torch.float32).to(device).unsqueeze(-1).permute(1,0,2)
    Y_val=torch.tensor(Y_val, dtype=torch.int64).to(device)
    times_val=torch.tensor(np.arange(X_val.shape[0])).float().to(device)
    times_val=times_val.unsqueeze(1).repeat(1,X_val.shape[1])
    
    val.append(X_val)
    val.append(times_val)
    val.append(Y_val)
    
    # test
    X_test=torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(-1).permute(1,0,2)
    Y_test=torch.tensor(Y_test, dtype=torch.int64).to(device)
    times_test=torch.tensor(np.arange(X_test.shape[0])).float().to(device)
    times_test=times_test.unsqueeze(1).repeat(1,X_test.shape[1])
    
    test.append(X_test)
    test.append(times_test)
    test.append(Y_test)
    
    print("\nFinal label ranges after processing:")
    print(f"Training labels: [{Y_training.min()}, {Y_training.max()}]")
    print(f"Validation labels: [{Y_val.min()}, {Y_val.max()}]")
    print(f"Test labels: [{Y_test.min()}, {Y_test.max()}]")
    
    if return_train_loader:
        return SynthTrainDataset_Batch_first(train[0],train[1],train[2]), val, test
    
    if return_dict:
        train_dataset = Get_torch_dataset(train[0],train[1],train[2],T_first=True)
        val_dataset = Get_torch_dataset(val[0],val[1],val[2],T_first=True)
        test_dataset = Get_torch_dataset(test[0],test[1],test[2],T_first=True)
        return train_dataset, val_dataset, test_dataset, None
    else:
        return train, val, test, None
    
    
def process_Synth_(split_no = 1, device = None, base_path = None, regression = False,
        label_noise = None):

    split_path = os.path.join(base_path, 'split={}.pt'.format(split_no))

    D = torch.load(split_path)

    D['train_loader'].X = D['train_loader'].X.float().to(device)
    D['train_loader'].times = D['train_loader'].times.float().to(device)
    if regression:
        D['train_loader'].y = D['train_loader'].y.float().to(device)
    else:
        D['train_loader'].y = D['train_loader'].y.long().to(device)

    val = []
    val.append(D['val'][0].float().to(device))
    val.append(D['val'][1].float().to(device))
    val.append(D['val'][2].long().to(device))
    if regression:
        val[-1] = val[-1].float()
    D['val'] = tuple(val)

    test = []
    test.append(D['test'][0].float().to(device))
    test.append(D['test'][1].float().to(device))
    test.append(D['test'][2].long().to(device))
    if regression:
        test[-1] = test[-1].float()
    D['test'] = tuple(test)

    if label_noise is not None:
        # Find some samples in training to switch labels:

        to_flip = int(label_noise * D['train_loader'].y.shape[0])
        to_flip = to_flip + 1 if (to_flip % 2 == 1) else to_flip # Add one if it isn't even

        flips = torch.randperm(D['train_loader'].y.shape[0])[:to_flip]

        max_label = D['train_loader'].y.max()

        for i in flips:
            D['train_loader'].y[i] = (D['train_loader'].y[i] + 1) % max_label

    return D