from td_features import *
from preprocess_task import *
from plotting import *

# Sample usage
if __name__ == "__main__":
    # Simulate an 8 channel signal with length 5000
    rng = np.random.default_rng(seed=42)
    arr = rng.random((8, 5000)) * 256 - 128 # random int8 values

    # Plot raw data
    create_plot(arr, y_range=(-128, 127), title="Sample Signal").show()

    # Windower object that creates windows with length 50 and overlap of 40
    wd = Windower(arr.shape[1], 50, 40)

    # SignalFilter object with a bandpass and notch filter
    ft = SignalFilter(200)
    ft.add_bandpass(5, 95)
    ft.add_notch(50)

    # Feature extractor with 3 vectorised functions and 1 non-vectorised function
    fe = TDExtractor()
    fe.add_vectorised_features([mav, wl, wamp_5])
    fe.add_features(ssc)

    # Create SignalPreprocessor manager object and add the above tasks
    pre = SignalPreprocessor()
    pre.add_tasks([wd, ft, fe])
    pre.setup_tasks() # setup all
    res = pre.process_tasks(arr) # run preprocessing pipeline
    print(res.shape)

    # Plot feature 0 over all channels
    create_plot(res[:, :, 0], y_range=(0, 127), title="Feature 0").show() 