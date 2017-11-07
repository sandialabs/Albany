import requests
from bs4 import BeautifulSoup
import sys
import re

not_configured_regex = 'SCOREC'
not_tested_regex = 'Cori|Edison|Trilinos|SCOREC|Peridigm'

sites = [
  'http://my.cdash.org/index.php?project=Albany',
  'http://cdash.sandia.gov/CDash-2-3-0/index.php?project=Albany']

for site in sites:
  page = requests.get(site)
  soup = BeautifulSoup(page.content, 'html.parser')
  past_subprojects = False
  for tag in soup.find_all('tr'):
    if past_subprojects:
      try:
        build_name = tag.td.a.get_text()
        children = list(tag.children)
        number_texts = []
        for i in range(2, 10):
          number_texts.append(children[2 * i - 1].get_text())
        if not re.search(not_configured_regex, build_name, re.IGNORECASE):
          if not number_texts[0].isdecimal():
            print("Configure not reported for", build_name)
          elif int(number_texts[0]) > 0:
            print(build_name, "had", int(number_texts[0]), "configure errors")
        if not number_texts[3].isdecimal():
          print("Build not reported for", build_name)
        elif int(number_texts[3]) > 0:
          print(build_name, "had", int(number_texts[3]), "compile errors")
        if None == re.search(not_tested_regex, build_name, re.IGNORECASE):
          if not number_texts[7].isdecimal():
            print("Tests not reported for", build_name)
          elif int(number_texts[7]) > 0:
            print(build_name, "had", int(number_texts[7]), "test failures")
      except AttributeError:
        pass
    else:
      try:
        if 'SubProjects' == tag.td.h3.get_text():
          past_subprojects = True
      except AttributeError:
        pass
