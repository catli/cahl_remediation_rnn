import numpy as np
import os
import pdb
import csv
import time

def read_vector_file(path, file_name):
    full_path = os.path.expanduser(path+file_name +'.npy')
    print(full_path)
    condense_data =  np.load(full_path)
    return condense_data


class SummarizeStuckness():

    def __init__(self, write_filename):
        self.condense_data = read_vector_file(path = '~/cahl_rnn_output/',
            file_name = 'condense_session_data')
        self.writefile = open(write_filename, 'w')
       
    def iterate_through_sessions(self):
        '''
            read file and write the lines
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''
        self.user_data = {'stuck':{},'unstuck':{}, 'never_stuck':[] }
        self.csvwriter = csv.writer(self.writefile, delimiter = ',') 
        self.write_header()
        counter = 1
        for session in self.condense_data:
            self.summarize_session(session)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
 
    def summarize_session(self, session):
        '''
           Parse through each line and store the values 
        '''
        self.user_attempts = {}
        for attempt in session:
            content_type = attempt[0]
            if content_type == 'exercise':
                self.add_to_user_attempts(attempt)
            self.add_to_stuck(attempt)
        self.summarize_user_data() 

    def add_to_user_attempts(self, attempt):
        '''
            Add correct or incorrect for problem 
        '''
        problem = attempt[1]
        correct = attempt[2] == 'true'
        if problem not in self.user_attempts:
            self.user_attempts[problem] = {}
            self.user_attempts[problem]['correct'] = 0
            self.user_attempts[problem]['incorrect'] = 0
        if correct:
            self.user_attempts[problem]['correct']+=1
        else:
            self.user_attempts[problem]['incorrect']+=1
        self.add_new_problem_for_user(attempt, problem)
 
    def add_new_problem_for_user(self, attempt, problem):
        '''
           summarize the problem 
        '''
        if self.user_attempts[problem]['correct']>=2 and problem in self.user_data['stuck']:
            self.user_data['unstuck'][problem]  = self.summarize_unstuck(problem)
            del self.user_data['stuck'][problem]  
        if problem in self.user_data['unstuck'] or  \
            problem in  self.user_data['never_stuck']:
            pass 
        elif self.user_attempts[problem]['correct']>=2 and \
            problem not in self.user_data['stuck']:
            self.user_data['never_stuck'].append(problem)
        elif self.user_attempts[problem]['incorrect']>=2 and \
            problem not in self.user_data['stuck']:
            self.user_data['stuck'][problem] = []

    def add_to_stuck(self, attempt):
        '''
           add the new problem, video, or hint to the stuck  
        '''
        content_type = attempt[0]
        content_name = attempt[1]
        for stuck_item in self.user_data['stuck']:
            content_suffix = ''
            if content_type in ('hint','exercise'):  
                exercise_name = content_name.split('|')[0] 
                stuck_exercise = stuck_item.split('|')[0]
                # whether the hint taken or problem practiced
                # belong to the same exercise as the
                # stuck problem type
                if stuck_exercise == exercise_name:
                    content_suffix = '_same_ex'
                else:
                    content_suffix = '_diff_ex'
            content_token = content_type + content_suffix  
            if stuck_item == content_name:
                content_token == 'same_problem'
            self.user_data['stuck'][stuck_item].append(content_token)


    def summarize_unstuck(self, problem):
        '''
            output: output the unstuck summary stats array
            which lists the attibutes of the unstuckness token
            ['different_exercise','remediation_problems','correct_remediation_problems']
        '''
        remediation_list = self.user_data['stuck'][problem]
        unstuck_state = {}
        unstuck_state['remediation_contents'] = len(remediation_list) 
        unstuck_state['remediation_videos'] = sum([item == 'video' for item in remediation_list]) 
        unstuck_state['remediation_hints_on_same_exercise'] = sum(
                [item == 'hint_same_ex' for item in remediation_list]) 
        unstuck_state['remediation_hints_on_diff_exercise'] = sum(
                [item == 'hint_diff_ex' for item in remediation_list]) 
        unstuck_state['remediation_problems_on_same_exercise'] = sum(
                [item == 'exercise_same_ex' for item in remediation_list]) 
        unstuck_state['remediation_problems_on_diff_exercise'] = sum(
                [item == 'exercise_diff_ex' for item in remediation_list]) 
        return unstuck_state

    def summarize_user_data(self):
        '''
            summarize user data 
            and write the array of summary stats
            to the file
        '''
        never_stuck_problems = len(self.user_data['never_stuck'])
        never_unstuck_problems = len(self.user_data['stuck'].keys())
        unstuck_problems = len(self.user_data['unstuck'].keys())
        unstuck_problems_with_videos = 0
        unstuck_problems_with_hints = 0
        unstuck_problems_with_practice = 0
        unstuck_problems_with_video_only = 0
        unstuck_problems_with_hint_only = 0
        unstuck_problems_with_practice_only = 0
        unstuck_num_contents = 0
        unstuck_num_videos = 0
        unstuck_num_hints_same_ex = 0
        unstuck_num_hints_diff_ex = 0
        unstuck_num_practice_same_ex = 0
        unstuck_num_practice_diff_ex = 0
        for unstuck_item in self.user_data['unstuck']:
            unstuck_state = self.user_data['unstuck'][unstuck_item] 
            any_video = int(unstuck_state['remediation_videos'] > 0 )
            any_hint = int( (unstuck_state['remediation_hints_on_same_exercise'] + 
                    unstuck_state['remediation_hints_on_diff_exercise'])>0)
            any_practice = int( (unstuck_state['remediation_problems_on_same_exercise'] + 
                    unstuck_state['remediation_problems_on_diff_exercise']) > 0 )
            unstuck_num_contents += unstuck_state['remediation_contents']  
            unstuck_num_videos += unstuck_state['remediation_videos'] 
            unstuck_num_hints_same_ex += unstuck_state['remediation_hints_on_same_exercise']
            unstuck_num_hints_diff_ex += unstuck_state['remediation_hints_on_diff_exercise']
            unstuck_num_practice_same_ex += unstuck_state['remediation_problems_on_same_exercise']
            unstuck_num_practice_diff_ex += unstuck_state['remediation_problems_on_diff_exercise']
            unstuck_problems_with_videos +=  any_video
            unstuck_problems_with_hints +=  any_hint
            unstuck_problems_with_practice += any_practice
            unstuck_problems_with_video_only += int(any_video and (any_hint+any_practice)==0)
            unstuck_problems_with_hint_only = int(any_hint and (any_video+any_practice)==0)
            unstuck_problems_with_practice_only = int(any_practice and (any_video+any_hint)==0)
        self.csvwriter.writerow([ 
            never_stuck_problems, 
            never_unstuck_problems,
            unstuck_problems,
            unstuck_num_contents, 
            unstuck_num_videos, 
            unstuck_num_hints_same_ex, 
            unstuck_num_hints_diff_ex, 
            unstuck_num_practice_same_ex,
            unstuck_num_practice_diff_ex,
            unstuck_problems_with_videos,
            unstuck_problems_with_hints,
            unstuck_problems_with_practice, 
            unstuck_problems_with_video_only, 
            unstuck_problems_with_hint_only,
            unstuck_problems_with_practice_only])
        # clear user data 
        self.user_data = {'stuck':{},'unstuck':{}, 'never_stuck':[]}

    def write_header(self):
        self.csvwriter.writerow([ 
            'never_stuck_problems', 
            'never_unstuck_problems',
            'unstuck_problems',
            'unstuck_num_contents', 
            'unstuck_num_videos', 
            'unstuck_num_hints_same_ex', 
            'unstuck_num_hints_diff_ex', 
            'unstuck_num_practice_same_ex',
            'unstuck_num_practice_diff_ex',
            'unstuck_problems_with_videos',
            'unstuck_problems_with_hints',
            'unstuck_problems_with_practice', 
            'unstuck_problems_with_video_only', 
            'unstuck_problems_with_hint_only',
            'unstuck_problems_with_practice_only'
        ])

def main():
    write_path = '~/cahl_rnn_output/summarize_stuckness_bysession.csv'
    write_file = os.path.expanduser(write_path)
    stuck = SummarizeStuckness(write_file)
    stuck.iterate_through_sessions()

if __name__ == '__main__':
    start = time.time() 
    main()
    end =time.time()
    print(end-start)
    

