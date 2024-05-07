import os
from glob import glob
import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from axoden.volume_projections import (
    remove_spines_plot, intensity_along_axis, remove_ticks_plot,
    collect_info_from_filename, generate_background_mask, collect_within_mask,
    convert_image_to_gray, binarize_image, count_pixels, compute_area,
    compute_threshold, load_table, save_table,
    process_folder, write_summary_data_plot, write_signal_intensity_along_axis_plot
)


def _sample_fig():
    fig, ax = plt.subplots(1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    return fig, ax


@pytest.mark.parametrize("loc",
    (
        ['top', 'right', 'bottom', 'left'],
        ['top', 'bottom'],
        ['left'],
    )
)
def test_remove_spines_plot(loc):
    _, ax = _sample_fig()

    loc_all = ['top', 'right', 'bottom', 'left']
    for l in loc_all:
        assert ax.spines[l].get_visible()
    remove_spines_plot(ax, loc)
    
    for l in loc:
        # if l in loc, then it should be invisible
        assert ax.spines[l].get_visible() != (l in loc)


def test_intensity_along_axis():
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0], # 0
            [0, 0, 0, 1, 0, 1], # 2
            [0, 1, 1, 0, 0, 1], # 3
            [1, 1, 1, 1, 1, 1], # 6
            #1, 2, 2, 2, 1, 3
        ]
    )

    expected_y = np.array([0, 2, 3, 6])
    expected_x = np.array([1, 2, 2, 2, 1, 3])
    
    # default, int
    res_x, res_y = intensity_along_axis(img)
    assert np.all(res_x == expected_x)
    assert np.all(res_y == expected_y)

    # default, float32
    res_x, res_y = intensity_along_axis(img.astype(np.float32))
    assert np.all(res_x == expected_x.astype(np.float32))
    assert np.all(res_y == expected_y.astype(np.float32))

    # ax=x
    res_x = intensity_along_axis(img, ax='x')
    assert np.all(res_x == expected_x)

    # ax=y
    res_y = intensity_along_axis(img, ax='y')
    assert np.all(res_y == expected_y)

    # error handling
    with pytest.raises(ValueError):
        intensity_along_axis(img, ax='wrong_value')


def test_remove_ticks_plot():
    _, ax = _sample_fig()

    # we should have ticks initially
    assert ax.get_xticks().shape[0] != 0
    assert ax.get_yticks().shape[0] != 0

    remove_ticks_plot(ax, loc='all')
    assert ax.get_xticks().shape[0] == 0
    assert ax.get_yticks().shape[0] == 0
    remove_ticks_plot(ax, loc='all')  # calling multiple times should work
    assert ax.get_xticks().shape[0] == 0
    assert ax.get_yticks().shape[0] == 0

    _, ax = _sample_fig()
    remove_ticks_plot(ax, loc='x')
    assert ax.get_xticks().shape[0] == 0
    assert ax.get_yticks().shape[0] != 0

    _, ax = _sample_fig()
    remove_ticks_plot(ax, loc='y')
    assert ax.get_xticks().shape[0] != 0
    assert ax.get_yticks().shape[0] == 0

    _, ax = _sample_fig()
    with pytest.raises(ValueError):
        remove_ticks_plot(ax, loc='wrong_Argument')

@pytest.mark.parametrize(("filename", "expected_animal", "expected_brain_area", "expected_group"),
    (
        ("0001_ACC_grp1.tif", "0001", "ACC", "grp1"),
        ("1.1_ACC.x_group.2.tif", "1.1", "ACC.x", "group.2"),
        ("animal1_acc.tif", "animal1", "acc", ""),
        ("folder/0001_ACC.tif", "0001", "ACC", ""),
        ("0001_ACC_groupx_unused.tif", "0001", "ACC", "groupx"),
        ("0001_ACC-SUB_grp-x.tif", "0001", "ACC-SUB", "grp-x"),
        ("0001_ACC-SUB_GRP1_unused_unused2.tif", "0001", "ACC-SUB", "GRP1"),
    )
)
def test_collect_info_from_filename(filename, expected_animal, expected_brain_area, expected_group):
    animal, brain_area, group = collect_info_from_filename(filename)

    assert animal == expected_animal
    assert brain_area == expected_brain_area

    if expected_group:
        assert group == expected_group
    else:
        assert group  # there should be a default value


def test_collect_info_from_filename_error():
    with pytest.raises(ValueError):
        collect_info_from_filename("animal.tif")


@pytest.mark.parametrize("dtype", (np.float32, np.int32))
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("is_masked", (False, True))
def test_generate_mask(dtype, dim, is_masked):
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0], # 0
            [0, 0, 0, 1, 0, 1], # 2
            [0, 1, 1, 0, 0, 1], # 3
            [1, 1, 1, 1, 1, 1], # 6
            #1, 2, 2, 2, 1, 3
        ]
    ).astype(dtype)
    expected_mask = img == 0

    if dim == 3:
        img = np.stack((img, img, img), axis=-1)
    elif dim == 2:
        pass  # already 2d
    else:
        raise ValueError('only dim 3 and dim 2 are tested here')

    mask = generate_background_mask(img, is_masked)
    assert mask.dtype == "bool"

    if is_masked:
        assert np.all(mask == expected_mask)
    else:
        assert np.all(mask == False)

    #PIL Image
    img_pil = Image.fromarray(img.astype(np.uint8))
    mask = generate_background_mask(img_pil, is_masked)
    assert mask.dtype == "bool"
    if is_masked:
        assert np.all(mask == expected_mask)
    else:
        assert np.all(mask == False)


def test_collect_within_mask():
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    ).astype(np.float32)
    
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.float32)

    expected_result = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, np.nan, np.nan, np.nan, np.nan, 1],
            [0, np.nan, np.nan, np.nan, np.nan, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    ).astype(np.float32)

    result = collect_within_mask(img, mask)
    np.testing.assert_equal(result, expected_result)


def test_convert_image_to_gray():
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 2],
            [0, 3, 4, 0, 0, 5],
            [6, 7, 8, 9, 1, 2],
        ]
    ).astype(np.uint8)
    img_pil = Image.fromarray(np.stack([img, img, img], axis=-1))
    img_gray = convert_image_to_gray(img_pil)
    assert img_gray.mode == 'L'

    img_gray = np.array(img_gray)
    assert np.all(img_gray == img)


@pytest.mark.parametrize("dtype", (np.uint8, np.float32))
@pytest.mark.parametrize("threshold", (0, 1.5, 0.5, 5, 100000))
def test_binarize_image(dtype, threshold):
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 2],
            [0, 3, 4, 0, 0, 5],
            [6, 7, 8, 9, 1, 2],
        ]
    ).astype(dtype)

    result = binarize_image(img, threshold)
    index = np.where(img > threshold)
    expected_result = np.zeros(img.shape, dtype=dtype)
    expected_result[index] = 1

    # ignoring dtype for testing
    assert np.all(result == expected_result)

@pytest.mark.parametrize("dtype", (np.uint8, np.int64, np.float32))
def test_count_pixels(dtype):
    img = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ]
    ).astype(dtype)
    expected_n_black = 12
    expected_n_white = 8
    expected_n_total = 20  # ignoring nans

    # test 2d binarized image
    n_white, n_black, n_total = count_pixels(img)
    assert n_white == expected_n_white
    assert n_black == expected_n_black
    assert n_total == expected_n_total

    # test 1d reshaped binarized image
    n_white, n_black, n_total = count_pixels(np.reshape(img, -1))
    assert n_white == expected_n_white
    assert n_black == expected_n_black
    assert n_total == expected_n_total

def test_count_pixels_error():
    img = np.array(
        [
            [2, 0, 1, 0, 0],  # 2 is not a binarized image
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ]
    ).astype(np.int64)

    with pytest.raises(ValueError):
        count_pixels(img)

    img = np.array(
        [
            [0.5, 0, 1, 0, 0], # 0.5 is not a binarized image
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ]
    ).astype(np.float32)

    with pytest.raises(ValueError):
        count_pixels(img)


@pytest.mark.parametrize("pixel_size", (1.0, 0.0, 0.75))
def test_compute_area(pixel_size):
    img = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ]
    ).astype(np.int64)
    n_white = 8
    n_black = 12
    n_total = 20  # ignoring nans
    area_single_pixel = pixel_size**2

    w, b, all = compute_area(img, pixel_size)
    assert w == n_white * area_single_pixel
    assert b == n_black * area_single_pixel
    assert all == n_total * area_single_pixel


def test_compute_threshold():
    img = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 0.2, 0, 1, 0],
            [0, 0.4, 0.4, 0, 0],
            [0.7, 0.8, 0, 1, 1],
        ]
    ).astype(np.float32)
    img *= 255
    img = img.astype(np.uint8)

    thr = compute_threshold(img)
    # using wide range here, we don't care about the exact value
    # we care only to get notified if something goes really wrong
    assert thr == pytest.approx(102, 10)


def test_load_table_data():
    expected_columns = sorted([
        'animal', 'brain_area', 'group',
        'pixels_signal', 'pixels_black', 'pixels_total',
        'threshold', 'area_image', 'area_signal', 'area_black', 'area_img_um',
        'percent_signal',
    ])

    # no index
    table = load_table("tests/data/data.csv")
    assert isinstance(table, pd.DataFrame)
    assert sorted(table.columns) == expected_columns

    # with index
    table_index = load_table("tests/data/data_index.csv")
    assert isinstance(table_index, pd.DataFrame)
    assert sorted(table_index.columns) == expected_columns

    assert np.all(table == table_index)


def test_load_table_data_axis():
    expected_columns = sorted([
        'animal', 'brain_area', 'group',
        'signal_bin_x_ax', 'signal_bin_y_ax',
        'signal_gray_x_ax', 'signal_gray_y_ax',
    ])

    # no index
    table = load_table("tests/data/data_axis.csv")
    assert isinstance(table, pd.DataFrame)
    assert sorted(table.columns) == expected_columns

    # with index
    table_index = load_table("tests/data/data_axis_index.csv")
    assert isinstance(table_index, pd.DataFrame)
    assert sorted(table_index.columns) == expected_columns

    # we don't compare the np arrays in each cell
    assert np.all(table[['animal', 'brain_area', 'group']] == table_index[['animal', 'brain_area', 'group']])

def test_save_table(tmpdir):
    table = load_table("tests/data/data.csv")
    filename_saved = "data_save.csv"
    path_saved = os.path.join(tmpdir, filename_saved)

    save_table(table, tmpdir, path_saved)
    table_saved = load_table(path_saved)
    assert np.all(table == table_saved)


def test_process_folder(tmpdir):
    pixel_size = 0.75521
    is_masked = True

    # before processing, there should be no csv files saved
    csv_files = glob(os.path.join(tmpdir, "*.csv"))
    assert(len(csv_files) == 0)

    data, data_axis = process_folder("tests/data", pixel_size, is_masked, output_folder=tmpdir, save=True)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(data_axis, pd.DataFrame)

    csv_files = glob(os.path.join(tmpdir, "*.csv"))
    # two csv files should be saved
    assert(len(csv_files) == 2)

    # now write the summary data plots
    pdf_files = glob(os.path.join(tmpdir, "*.pdf"))
    assert len(pdf_files) == 0

    write_summary_data_plot(tmpdir, data)
    pdf_files = glob(os.path.join(tmpdir, "*.pdf"))
    assert len(pdf_files) == 1

    write_signal_intensity_along_axis_plot(tmpdir, data_axis, pixel_size)
    pdf_files = glob(os.path.join(tmpdir, "*.pdf"))
    assert len(pdf_files) == 2
