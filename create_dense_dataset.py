import time 
import csv
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
    return sha_id + '|' + session_start

class CreateLearningList():

    def __init__(self, read_filename):
        print('initialize '+ read_filename)
        self.reader = open(read_filename,'r')
        self.learning_list = [] 
        self.iterate_through_lines(sessions = True)
        #[TODO]: WRITE LEARNING LIST  

    def iterate_through_lines(self, sessions = False):
        '''
            read file and write the lines
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''
        self.last_sha_id = 'sha_id'
        self.last_problem = 'exercise|problem_type'
        self.user_attempts = {}
        self.user_data = {'stuck':{},'unstuck':{}, 'never_stuck':[] }
        self.csvwriter = csv.writer(self.writefile, delimiter = ',') 
        self.write_header()
        next(self.reader)
        # first_line = self.reader.readline()
        counter = 1
        for line in self.reader:
            self.parse_line(line, sessions)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)

    def parse_line(self, line, sessions=False):
        '''
           Parse through each line and store the values 
        '''
        line_delimited = line.split(',')
        # if sessions = True, then id by session
        if sessions:
            sha_id = line_delimited[0]+line_delimited[2]
        else:
            sha_id = line_delimited[0]
        # if sha_id already in learning list, then skip
        if sha_id in self.learning_list:
            continue
        exercise = line_delimited[5]
        problem_type = line_delimited[7]
        correct = line_delimited[8] == 'true'
        attempt_numbers = int(line_delimited[12])
        problem = exercise + '|' +  problem_type
        if sha_id != self.last_sha_id:
            self.summarize_old_sha_id(prerequisites)
            self.last_sha_id = sha_id
            self.last_problem = problem
            self.user_attempts = {}
            self.update_attempts(correct, attempt_numbers, problem) 
        else:
            self.update_attempts(correct, attempt_numbers, problem)  
            self.add_new_data_for_user(problem_type, exercise)
            self.last_problem = problem
 

    def update_attempts(self, correct, attempt_numbers, problem):
        if problem not in self.user_attempts:
            self.user_attempts[problem] = {}
            self.user_attempts[problem]['correct'] = 0
            self.user_attempts[problem]['incorrect'] = 0
        if correct:
            self.user_attempts[problem]['correct']+=1
        else:
            self.user_attempts[problem]['incorrect']+= max(attempt_numbers-1,1)
             
    def add_new_data_for_user(self, problem_type, exercise):
        problem = exercise + '|' + problem_type 
        if self.user_attempts[problem]['correct']>=2 and problem in self.user_data['stuck']:
            # If got a problem right twice (which they were previously stuck on
            # then move the problem to the stuck list
            self.user_data['unstuck'].append(problem)
            # if unstuck, then add to list of unstuck sha_ids
            self.learning_list.append(sha_id)
        elif self.user_attempts[problem]['correct']>=2 and \
            problem not in self.user_data['stuck']:
            self.user_data['never_stuck'].append(problem)
        elif self.user_attempts[problem]['incorrect']>=2 and problem not in self.user_data['stuck']:
            self.user_data['stuck'].append(problem)




class TokenizeData():
    '''
        create the dense data representing activity per session
        for learner. Working on exercises, watching a video and taking hints
        are all included in hte dense dataset 
    '''
    def __init__(self, exercise_filename, video_filename, learning_list):
        self.store_video_data(video_filename)
        self.session_data = [['NULL']
        self.session_index = [['NULL']]
        print('initialize '+ exercise_filename)
        self.exercise_reader = open(exercise_filename,'r')

    #[TODO: ADD VIDEO! FUNCTION TO READ DATA INTO DICTIONARY, i.e. 
    # { sha_id|session_start_time: [(timestamp, video), (timestamp, video),
    # ...]}
    def store_video_data(self, video_filename): 
        self.video_data = {}
        with reader as open(video_filename, 'r'):
            for line in reader:
                row = line.split(",")
                # session concatenates sha_id and session start time
                sha_id = row[0] 
                session_start = row[1] 
                session = create_session_id(sha_id , session_start)
                video_id = row[2]
                # start time for individual video
                start_time = row[6]
                if session not in self.video_data:
                    self.video_data[session] = [(start_time, video_id)]
                else:
                    self.video_data[session].append((start_time, video_id))
                 

    # [TODO] CREATE SESSION DATA 
    def create_session_data(self):
        '''
            generate a token for each row of data
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''
        counter = 1
        for line in self.reader:
            row = line.split(",")
            # [TODO] VIDEO! CHECK TO SEE IF THERE'S ANY RELEVANT VIDEOS BEFORE
            # [TODO] HINT! CHECK TO SEE IF THERE SHOULD BE ANY HINT STORED
            # AFTER EXERCISE ATTEMPT
            self.append_data(row)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
        self.delete_header()
        print('generated lines of token: '+ str(len(self.session_data)))
        
    def delete_header(self):
        self.session_data = self.session_data[2:]
        self.session_index = self.session_index[2:]
        
    def append_data(self, row):
        row_index = self.create_index(row)
        condensed_exercise = self.create_condense_exercise(row)
        session = create_session_id(row[0], row[2])
        start_time = row[3]
        hints = row[10]
        # [TODO] ADD THE VIDEOS! WATCHED BEFORE EXERCISE
        videos_watched_before = self.list_videos_watched_before(
                session = session, 
                exercise_start_time = start_time)
         
        if row_index == self.session_index[-1][0]:
            '''if the session matches the last session store
            append new row token to the last array in tokenize data'''
            for condensed_video in videos_watched_before:
                self.session_data[-1].append(condensed_video)
            self.session_data[-1].append(condensed_exercise)
        elif row_index != self.session_index[-1]:
            '''
            If the index does not match, then store as new session
            '''
            self.session_index.append([row_index])
            self.session_data.append([row_token])
        # [TODO] TRANSFORM THE NUMBER OF HINTS! TAKEN TO CONDENSE
   
   def create_index(self, row):
        '''
            the index for each entry is based on individual sessions 
            and unique id for session combines sha_id and session_start_time
            create a tuple with these two values 
        '''
        session_index = (row[0], row[2])
        return(session_index)

    def create_condense_exercise(self, row):
        '''
            create a token for learner entry
            concatenates the exercise, problem type, and correctness value
        '''
        exercise = row[5]
        problem_type = row[7].replace(' ','')
        content_name = exercise+'|'+ problem_type
        response = row[8]
        return(('exercise',content_name, response))

    def list_videos_watched_before(self, session, exercise_start_time:
        '''
           Create a list of videos watched before exercise start time
        '''
        videos_watched_before = []
        #[TODO] CREATE LIST OF VIDEOS!  WATCHED BEFORE EXERCISE
        if session in self.video_data: 
            # if session in video watch list, extract list
            # iterate for each video
            session_videos = self.video_data[session]
            for i, video_instance in enumerate(session_videos):
                video_timestamp = video_instance[0]
                if video_instance < exercise_start_time:
                    # if occurred before exercise, then create condense
                    # video data and delete, otherwise
                    video_id = video_instance[1]
                    videos_watched_before.append(('video',video_id, ))
                    del self.video_data[session][i-1] 
                else:
                    # otherwise end iteration
                    break
        return videos_watched_before 

    # [TODO] ADD FUNCTION TO SUMMARIZE THE HINTS TAKEN DATA 

       
def write_file(file_name, data_array):
    '''write file. Estimated time: 1.5 sec for 1M rows'''
    path = 'sorted_data/'+file_name+'.csv'
    print(path)
    open_file = open(path, "w")
    with open_file:
        csvwriter = csv.writer(open_file, delimiter = ' ')
        csvwriter.writerows(data_array)


# [TODO] RENAME ALL FILES FROM TOKENIZE
def generate_token_files(affix):
    '''
        Read the file name given by the affix, assume file with khan_data_<affix> exists in
        sorted_data directory. Generate tokenize data and the session index for each row.
        The tokens are represents unique exericse + problem_type + correctness. 
        Estimated time: 5 seconds for 1M row
    '''
    # create file names
    exercise_filename = 'sorted_data/khan_data_'+affix+'.csv'
    # [TODO]: UPDATE VIDEO! FILENAME
    video_filename = 'sorted_data/khan_video_data_'+affix+'.csv'
    # Create the list of sessions where learning occurred 
    learning_list = CreateLearningList(read_filename = exercise_filename).learning_list
    pdb.set_trace()
    #write_data_filename = 'tokenize_data_'+affix
    #write_index_filename = 'tokenize_index_'+affix
    ## generate token data
    #token = TokenizeData(exercise_filename, video_filename, learning_list)
    #token.create_session_data()
    ## write the index and dat files
    #write_file(write_data_filename, token.session_data)
    #write_file(write_index_filename, token.session_index)
    
     
affix = 'sorted'

start = time.time() 
generate_token_files(affix)
end =time.time()
print(end-start)

# Without logic for out of sort 
# tiny: 0.007
# small: 3.850081443786621 

# Adding save file 
# small: 5.166566610336304
# all: 8577.662670850754

# token.session_data
# token.session_index
    