import unittest
import random
import string
import os
import shutil

from youtube8m.download_video_ids import delete_category, is_finished, read_mids, ref_ids
from settings import YOUTUBE8M_CATEGORIES_URL

class TestDownloadVideoIds(unittest.TestCase):

    def setUp(self):
        # Create categories file
        self.categ_file = ''.join(random.choices(string.ascii_lowercase, k=10))
        file = open(self.categ_file, "w")
        file.write('mid = "03bt1gh"	Games (788288)\n')
        file.write('mid = "01mw1"	Video game (539945)\n')
        file.write('mid = "07yv9"	Vehicle (415890)\n')
        file.close()

        self.ids_file = ''.join(random.choices(string.ascii_lowercase, k=10))
        file = open(self.ids_file, "w")
        file.write('Games, lkjio\n')
        file.write('Video game, abcde\n')
        file.write('Vehicle, fhg\n')
        file.close()

    def tearDown(self):
        os.remove(self.categ_file)
        os.remove(self.ids_file)

    def test_read_mids(self):
        mids, categories = read_mids(self.categ_file)
        self.assertListEqual(categories, ["Games", "Video game", "Vehicle"])

    def test_read_mids2(self):
        mids, categories = read_mids(self.categ_file)
        self.assertListEqual(mids, ["03bt1gh", "01mw1", "07yv9"])

    def test_delete_category(self):
        ids_file2 = ''.join(random.choices(string.ascii_lowercase, k=10))
        shutil.copy(self.ids_file, ids_file2)
        delete_category("Vehicle", ids_file2)

        # Read file with deleted category
        file = open(ids_file2, "r")
        lines = file.readlines()
        file.close()

        os.remove(ids_file2)
        self.assertTrue(lines[-1].startswith("Video game"))

    def test_is_finished(self):
        finished, last_category = is_finished("Vehicle", self.ids_file)
        self.assertTrue(finished)

    def test_is_finished2(self):
        finished, last_category = is_finished("Video game", self.ids_file)
        self.assertFalse(finished)

    def test_ref_ids(self):
        mid = '01vzvy'
        mid_url = f"{YOUTUBE8M_CATEGORIES_URL}/{mid}.js"
        ids = ref_ids(mid_url)

        self.assertTrue(len(ids) > 0)

    def test_ref_ids2(self):
        mid = 'made_up_mid'
        mid_url = f"{YOUTUBE8M_CATEGORIES_URL}/{mid}.js"
        ids = ref_ids(mid_url)

        self.assertTrue(ids is None)