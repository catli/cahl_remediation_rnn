import os
import time 
import csv
import numpy as np
import pdb

############################################
# Tokenizes anonymized and sorted Khan Academy problem attempt data 
# to build word2vec model 
# Because of the dataset size (~200GB), it presorted
# using unix functions reference: 
# https://docs.google.com/document/d/1AbVJ78EP4HcUpxqo11XfuZh_vuga0UVpU70SiHgbiXA/edit#bookmark=id.r46be5wey369
# 


# [TODO] FILTER! OUT DATA WHERE LEARNERS NEVER GET STUCK
# AND THOSE WHERE LEARNERS NEVER UNSTUCK
# (FROM SUMMARIZE_STUCK FILE)

def create_session_id(sha_id, session_start):
    ''' concat the learner id with the session start time to get unique
    session id'''
    return str(sha_id) + '|' + str(session_start)

class CreateLearningList():

    def __init__(self, read_filename = ''):
        print('initialize '+ read_filename)
        if read_filename!='':
            self.reader = open(read_filename,'r')
        self.last_sha_id = 'sha_id'
        self.last_session_id = 'session_id'
        self.user_attempts = {}
        self.learning_list = set() 

    def iterate_through_lines(self):
        '''
            read file and write the lines
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''
        first_line = self.reader.readline()
        counter = 1
        for line in self.reader:
            line_delimited = line.split(',')
            session_id = create_session_id( line_delimited[0], 
                    line_delimited[2])
            # if sha_id already in learning list, then skip
            if session_id in self.learning_list:
                continue
            else:
                self.parse_line(line_delimited, session_id)
                counter+=1
                if counter % 1000000 == 0:
                    print(counter)

    def parse_line(self, line_delimited, session_id):
        '''
           Parse through each line and store the values 
        '''
        sha_id = line_delimited[0]
        exercise = line_delimited[5]
        problem_type = line_delimited[7]
        correct = line_delimited[8] == 'true'
        attempt_numbers = int(line_delimited[12])
        problem = exercise + '|' +  problem_type
        if session_id != self.last_session_id:
            self.last_session_id = session_id
            self.user_attempts = {}
            self.update_attempts(correct, attempt_numbers, problem, session_id) 
        else:
            self.update_attempts(correct, attempt_numbers, problem, session_id)  
 

    def update_attempts(self, correct, attempt_numbers, problem, session_id):
        if problem not in self.user_attempts:
            self.user_attempts[problem] = {}
            self.user_attempts[problem]['correct'] = 0
            self.user_attempts[problem]['incorrect'] = 0
        if correct:
            self.user_attempts[problem]['correct']+=1
        else:
            self.user_attempts[problem]['incorrect']+= max(attempt_numbers-1,1)
        if correct and self.user_attempts[problem]['correct']>=2 and \
            self.user_attempts[problem]['incorrect']>=2:
            self.learning_list.add(session_id)

    def test_learning_list(self):
        '''
            read file and write the lines
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''
        test_data = [
                # learner gets stuck and then unstuck
                'learner1,,2018-01-01,,,ex1,,p1,false,,,,2,1',
                'learner1,,2018-01-01,,,ex1,,p1,false,,,,2,1',
                'learner1,,2018-01-01,,,ex1,,p2,true,,,,1,1',
                'learner1,,2018-01-01,,,ex1,,p1,true,,,,1,1',
                'learner1,,2018-01-01,,,ex1,,p1,true,,,,1,1',
                'learner1,,2018-01-01,,,ex2,,p2,false,,,,2,1',
                # learner never get stuck
                'learner1,,2018-02-01,,,ex1,,p1,true,,,,1,1',
                'learner1,,2018-02-01,,,ex1,,p1,true,,,,1,1',
                'learner1,,2018-02-01,,,ex1,,p1,false,,,,3,1']
        start = time.time() 
    
        for line in test_data:
            line_delimited = line.split(',')
            session_id = create_session_id( line_delimited[0], 
                    line_delimited[2])
            # if sha_id already in learning list, then skip
            if session_id in self.learning_list:
                continue
            else:
                self.parse_line(line_delimited, session_id)
        # expect self.learning_list to equal to ['learn1|2018-01-01']
        assert self.learning_list == set([
                create_session_id('learner1','2018-01-01')])
        print('PASS TEST IN ')
        end =time.time()
        print(end-start)
   




class CondenseLearningData():
    '''
        create the dense data representing activity per session
        for learner. Working on exercises, watching a video and taking hints
        are all included in the dense dataset 
    '''
    def __init__(self ):
        self.session_data = [['NULL']]
        self.session_index = [['NULL']]

 
    def create_session_data(self, exercise_filename, 
            video_filename, learning_list):
        '''
            Interweave exercise and video activity into condense
            session data 
            generate a condense array for each row of data
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''
        print('initialize '+ exercise_filename)
        counter = 1
        exercise_reader = open(exercise_filename,'r')
        self.read_video_data(video_filename)
        for line in exercise_reader:
            row = line.split(",")
            if self.check_in_learning_list(row, learning_list):
                self.append_condense_data(row)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
        self.delete_header()
        print('generated number of sessions: %s' % str(len(self.session_data)))
    
  
    def read_video_data(self, video_filename):
        '''
           Create dictionary of videos watched for each session
            in this format: 
            { sha_id|session_start_time: [(timestamp, video), (timestamp, video),
            ...]}
        '''
        self.video_data = {}
        with open(video_filename, 'r') as video_reader:
            for line in video_reader:
                row = line.split(',')
                # session concatenates sha_id and session start time
                sha_id = row[0] 
                session_start = row[1] 
                session = create_session_id(sha_id , session_start)
                # unique video id
                video_id = row[4]
                # start time for individual video
                start_time = row[3]
                if session not in self.video_data:
                    self.video_data[session] = [(start_time, video_id)]
                else:
                    self.video_data[session].append((start_time, video_id))
                 
    def check_in_learning_list(self, row, learning_list):
        session_id = create_session_id(row[0], row[2])
        is_learner = session_id in learning_list
        return is_learner
    
    def append_condense_data(self, row):    
        '''
           Parse line from exercise, trnasform into condensed data
           and then append to session_data and session_index.
        '''
        session_id = create_session_id(row[0], row[2])
        start_time = row[3]
        exercise = row[5]
        problem_type = row[7].replace(' ','')
        content_name = exercise+'|'+ problem_type
        response = row[8]
        hints = int(row[10])
        condensed_exercise = self.create_condense_exercise(content_name, response)
        self.append_data(condensed_exercise, session_id, start_time, hints)
 
    def create_condense_exercise(self, content_name, response):
        '''
            Create a token for learner entry
            concatenates the exercise, problem type, and correctness value
        '''
        return(('exercise',content_name, response))

       
    def append_data(self, condensed_exercise,  session_id, start_time, hints):
        '''
           Append all relevant video, exercise and hints as condensed
           data in the session data
        '''
        videos_watched_before = self.list_videos_watched_before(
                session = session_id, 
                exercise_start_time = start_time)
        if session_id == self.session_index[-1][0]:
            '''
            If the session matches the last session store
            append new row token to the last array in tokenize data
            '''
            for condensed_video in videos_watched_before:
                self.session_data[-1].append(condensed_video)
            self.session_data[-1].append(condensed_exercise)
        elif session_id != self.session_index[-1]:
            '''
            If the index does not match, then store as new session
            '''
            self.session_index.append([session_id])
            self.session_data.append([condensed_exercise])
        if hints > 0:
            self.create_hints_data(condensed_exercise, hints)
   
    def list_videos_watched_before(self, session, exercise_start_time):
        '''
           Create a list of videos watched before exercise start time
           condensed video data has the format ('video', video_id)
        '''
        videos_watched_before = []
        if session in self.video_data: 
            # if session in video watch list, extract list
            # iterate for each video
            session_videos = self.video_data[session]
            for i, video_instance in enumerate(session_videos):
                video_timestamp = video_instance[0]
                if video_timestamp < exercise_start_time:
                    # if occurred before exercise, then create condense
                    # video data and delete, otherwise
                    video_id = video_instance[1]
                    videos_watched_before.append(('video',video_id, ))
                    del self.video_data[session][i-1] 
                else:
                    # otherwise end iteration
                    break
        return videos_watched_before 

    def create_hints_data(self, condensed_exercise, hints):
        '''
           Add the hints taken for each exercise with individual tokens
           condensed hint data has format: ('hints', content_name + hint_num, )
        '''
        for hint_num in range(hints):
            hint_id = condensed_exercise[1] + '|' + str(hint_num+1)
            self.session_data[-1].append(('hint', hint_id, ))

    def delete_header(self):
        self.session_data = self.session_data[1:]
        self.session_index = self.session_index[1:]

    def test_create_session_data(self):
        '''
            Test create_session_data behaves as expected
        '''
        learning_list = set(['learner1|2018-01-01'])
        self.video_data = {'learner1|2018-01-01':[('2018-01-01 2:00','video1')]}
        test_reader = [
                'learner1,,2018-01-01,2018-01-01 1:00,,ex1,,type 1,true,,0',
                'learner1,,2018-01-01,2018-01-01 3:00,,ex1,,type 2,false,,2',
                'learner1,,2018-02-01,2018-02-01 2:00,,ex3,,type 3,true,,0']
        counter = 1
        for line in test_reader:
            row = line.split(",")
            if self.check_in_learning_list(row, learning_list):
                self.append_condense_data(row)
        self.delete_header()
        assert self.session_index == [['learner1|2018-01-01']]
        assert self.session_data == [[
            ('exercise','ex1|type1','true'),('video','video1',),
            ('exercise','ex1|type2','false'),
            ('hint','ex1|type2|1',),('hint','ex1|type2|2',)
            ]]
        print('PASS TEST!')
        

       
def write_vector_file(path, file_name, vectors):
    path = os.path.expanduser(path+file_name+'.out')
    print(path)
    np.savetxt(path, vectors, delimiter = ',')

def write_set(path, file_name, writing_set):
    full_path = os.path.expanduser(path +file_name+'.csv')
    print 'writing set to %s' % (full_path)
    with open(full_path, "w") as open_file:
        for set_item in writing_set: 
            open_file.write(set_item + '\n')

def read_set(path, file_name):
    full_path = os.path.expanduser(path +file_name+'.csv')
    print 'reading set from %s' % (full_path)
    output_set = set()
    with open(full_path, "r") as reader:
        for line in reader: 
            output_set.add(line.strip())
    return output_set

def create_learning_list(affix):
    '''
        Create a list of sessions where learning occurred
        this list will be used to filter out non-learners from sample
    '''
    # create file names
    exercise_filename = os.path.expanduser(
        '~/sorted_data/khan_data_'+affix+'.csv')
   # Create the list of sessions where learning occurred 
    learning_list_instance = CreateLearningList(read_filename = exercise_filename)
    learning_list_instance.iterate_through_lines()
    write_set(path = '~/cahl_rnn_output/', 
            file_name = 'learning_list',
            writing_set = learning_list_instance.learning_list)
 

# [TODO] RENAME ALL FILES FROM TOKENIZE
def generate_token_files(affix):
    '''
        Read the file name given by the affix, assume file with 
        khan_data_<affix> exists in sorted_data directory. 
        Generate tokenize data and the session index for each row.
        The tokens are represents unique exericse + problem_type + correctness. 
        Estimated time: 5 seconds for 1M row
    '''
    # create file names
    exercise_filename = os.path.expanduser(
        '~/sorted_data/khan_data_'+affix+'.csv')
    video_filename = os.path.expanduser(
        '~/sorted_data/khan_video_data_'+affix+'.csv')
    # create_learning_list
    learning_list = read_set(path = '~/cahl_rnn_output/', 
            file_name = 'learning_list'  )
    ## generate condense data
    condense_data = CondenseLearningData()
    condense_data.create_session_data( exercise_filename, 
            video_filename, learning_list)
    ## write the index and data files
    write_vector_file(path = '~/cahl_rnn_output/', 
            file_name = 'condense_session_data', 
            vectors = condense_data.session_data)
    write_set(path = '~/cahl_rnn_output/', 
            file_name = 'condense_session_index',
            writing_set = condense_data.session_index)
    

def main():
    affix = 'sorted'
    start = time.time() 
    generate_token_files(affix)
    end =time.time()
    print(end-start)
   

if __name__ == '__main__':
    main() 
   
# Without logic for out of sort 
# tiny: 0.007
# small: 3.850081443786621 

# Adding save file 
# small: 5.166566610336304
# all: 8577.662670850754

# token.session_data
# token.session_index
    
