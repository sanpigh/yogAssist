from types import ClassMethodDescriptorType
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from os import listdir, remove
from os.path import isfile, join
import glob

URL_YJ_POSES = "https://www.yogajournal.com/poses/types/"
URL_YJ_REDIRECT = 'https://www.yogajournal.com/poses/'


class YogaJournalScraper():

    def parse(self,url):
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        return soup

    def get_types(self,soup):
        try:
            l_single_pose = soup.find_all(class_="c-article-lead u-breakout")
            if len(l_single_pose) == 1:
                sanskrit_name = soup.find("h3",text = "Sanskrit Name")
                sanskrit_name = sanskrit_name.next_sibling.next_sibling.text if sanskrit_name is not None else ''
                return [(soup.find(class_="c-article-headings u-spacing u-center-block").find("h1").text,
                        l_single_pose[0].find(class_="lazy")["data-iesrc"],
                        l_single_pose[0].find(class_="lazy")["data-iesrc"],
                        sanskrit_name)]

            l_types = soup.find_all(
                class_="c-block c-block-standard u-align--center c-block--inline-until-medium c-block--inline--mini")
            if len(l_types) == 0:
                l_types = soup.find_all(
                    class_="c-block c-block-standard u-align--center c-block--inline-until-medium")
            l_allPose = []
            for type_pose in l_types:
                dic_data = json.loads(type_pose.find(
                    class_='o-heading-link')['data-analytics-data'])['props']
                l_allPose.append((dic_data["title"],
                                  dic_data["path"],
                                  type_pose.find(class_="lazy")["data-iesrc"],
                                ''))
            return l_allPose
        except:
            return None

    def launch_scraping(self):
        soup = self.parse(URL_YJ_POSES)
        l_allPose = self.get_types(soup)
        all_pose = []
        while len(l_allPose) > 0:
            pose_type = l_allPose.pop()
            url = pose_type[1]
            r = requests.get(url)
            if r.url == URL_YJ_REDIRECT:
                all_pose.append((pose_type[0],pose_type[0].split('|')[-1], pose_type[2],'',url))
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            l_sub_types = self.get_types(soup)
            if l_sub_types is None:
                continue
            if len(l_sub_types) == 1:
                all_pose.append((pose_type[0], l_sub_types[0][0], l_sub_types[0][1],l_sub_types[0][3],url))
                continue
            else:
                if len(l_sub_types) == 1:
                    continue
                for sub_type in l_sub_types:
                    l_allPose.append(
                        (pose_type[0] + '|' + sub_type[0], sub_type[1], sub_type[2],sub_type[3],url))

        self.parsed_df = pd.DataFrame(all_pose)
        self.parsed_df.columns = [
            'classif', 'pose_name', 'pose_url', 'sanskrit name', 'url2'
        ]

        dic_files = self.get_yoga82_structure()
        l_links = self.get_clean_links(dic_files)
        self.write_links(l_links)
        return self

    def get_yoga82_structure(self):
        my_path = "raw_data/yoga_dataset_links"
        onlyfiles = [f"{f}" for f in listdir(my_path) if isfile(join(my_path, f))]
        dic_files = {}
        for x in onlyfiles:
            poses = x.replace('.txt', '')
            poses = poses.replace('_', ' ')
            poses = poses.split(' or ')
            for pose in poses:
                dic_files[pose.strip().lower().replace(' pose', '')] = x

        return dic_files

    def find_correspondant_filename(self,dic_files,english_name, sanskrit_name):
        english_name = english_name.strip().lower().replace(' pose' ,'')
        sanskrit_name = sanskrit_name.strip().lower()
        if english_name in dic_files:
            return dic_files[english_name]

        if sanskrit_name != '' and sanskrit_name in dic_files:
            return dic_files[sanskrit_name]
        for key in dic_files.keys():
            if sanskrit_name != '' and (key in sanskrit_name or sanskrit_name in key):
                return dic_files[key]
            if key in english_name or english_name in key:
                return dic_files[key]

        return ''

    def get_clean_links(self, dic_files):

        dico_map = {
          'Plank Pose – Step by Step Instructions' : 'Plank Pose',
          'King Pigeon Pose' : 'Pigeon Pose',
          'Cow Pose' : 'Cat Cow Pose',
          "One-Legged King Pigeon Pose II" : 'Rajakapotasana',
          'Half Frog Pose' : 'Frog Pose',
          'Revolved Triangle Pose' : 'Extended Revolved Triangle',
          'Pose Dedicated to the Sage Koundinya I' : 'Eka Pada Koundinyanasana I and II',
          "Bharadvaja’s Twist" : "Bharadvaja's_Twist_pose",
          'Revolved Side Angle Pose' : 'utthita padangusthasana',
          'Handstand or Downward Facing Tree Pose': 'Adho Mukha Vrksasana',
          'Extended Hand-to-Big-Toe Pose' : 'Utthita Padangusthasana',
          'Monkey Pose' : 'split pose',
          'Lotus Pose':'Sitting pose 1 (normal)',
          'High Lunge, Crescent Variation' : 'Virabhadrasana I'}


        self.parsed_df['pose_name_rework'] = self.parsed_df['pose_name'].apply(
            lambda x: dico_map[x] if x in dico_map else x)
        self.parsed_df['file_link'] = self.parsed_df[[
            'pose_name_rework', 'sanskrit name'
        ]].apply(lambda x: self.find_correspondant_filename(dic_files, *x),
                 axis=1)

        manual_add = [
            [
                'Scorpion_pose_or_vrischikasana.txt',
                "https://www.yogajournal.com/wp-content/uploads/2012/10/e1.jpg?crop=535:301&width=1070&enable=upscale"
            ],
            [
                'Upward_Plank_Pose_or_Purvottanasana_.txt',
                "https://www.yogajournal.com/wp-content/uploads/2007/08/upward-plank-pose.jpg?crop=1:1&width=250&enable=upscale"
            ],
            [
                'Supta_Virasana_Vajrasana.txt',
                "https://www.yogajournal.com/wp-content/uploads/2008/06/cleanse_278_4_fnlreclining-hero-pose-supta-virasana.jpg?crop=535:301&width=1070&enable=upscale"
            ],
            [
                'Tree_Pose_or_Vrksasana_.txt',
                "https://www.yogajournal.com/wp-content/uploads/2020/12/ccd03542-1.jpg?crop=535:301&width=1070&enable=upscale"
            ],
            [
                'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_.txt',
                "https://www.yogajournal.com/wp-content/uploads/2020/12/2hp_290_1721_bjk.jpg?crop=535:301&width=1070&enable=upscale"
            ],
            [
                "Cockerel_Pose.txt",
                "https://www.yogajournal.com/wp-content/uploads/2018/06/ty-landrum-9.jpg?crop=535:301&width=1070&enable=upscale"
            ]
        ]

        return self.parsed_df.loc[
            self.parsed_df['file_link'] != '',
            ['file_link', 'pose_url']].values.tolist() + manual_add
    def clean_folder_links(self, path):
        files_links = glob.glob(join(path,'*'))
        for f in files_links:
            remove(f)

    def write_links(self,data_yoga_journal):
        i = 1
        self.clean_folder_links('raw_data/yoga_journal_links')
        for img in data_yoga_journal:
            with open(join('raw_data/yoga_journal_links',img[0]),'a') as fy:
                fy.write(f'{img[0].replace(".txt","")}/yogajournal_{i}.jpg\t{img[1]}')
            i+=1


if __name__=='__main__':
    yj_scraper = YogaJournalScraper()
    yj_scraper.launch_scraping()
