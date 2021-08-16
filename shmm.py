import numpy as np
import mmap
from posix_ipc import Semaphore, O_CREX, ExistentialError, O_CREAT, SharedMemory, unlink_shared_memory
from ctypes import sizeof, memmove, addressof, create_string_buffer
from ctypes import Structure, c_int32, c_int64
from time import sleep

class MD(Structure):
    _fields_ = [
        ('shape_0', c_int32),
        ('shape_1', c_int32),
        ('shape_2', c_int32),
        ('size', c_int64),
        ('count', c_int64)
    ]


# Meta data in C struct { int shape_0, shape_1, shape_2; long size, count; }
#md_buf = create_string_buffer(sizeof(MD))


class ShmWrite:
    def __init__(self, name):
        self.shm_region = None

        self.md_region = SharedMemory(name + '-meta', O_CREAT, size=sizeof(MD))    # create shared memory to hold the MD struct
        self.md_buf = mmap.mmap(self.md_region.fd, self.md_region.size)            # pointing md_buf to the meta data (MD) shared memory
        self.md_region.close_fd()                                                  # release the corresponding file descriptor (not used)

        self.shm_buf = None
        self.shm_name = name    #save the shared memory name (ID)
        self.count = 0

        try:
            self.sem = Semaphore(name, O_CREX)    # create semaphore for sync access of shared memory (return error if already exist)
        except ExistentialError:
            sem = Semaphore(name, O_CREAT)    # dummy create
            sem.unlink()                      # remove the dummy
            self.sem = Semaphore(name, O_CREX) # create again
        self.sem.release()    # ensure it is not locked

    def add(self, frame: np.ndarray):    # put frame data in shared memory
        byte_size = frame.nbytes
        if not self.shm_region:
            self.shm_region = SharedMemory(self.shm_name, O_CREAT, size=byte_size)    # create shared memory to hold image data
            self.shm_buf = mmap.mmap(self.shm_region.fd, byte_size)                   # pointing shm_buf to image shared memory
            self.shm_region.close_fd()                                                # release the corresponding file descriptor (not used)

        self.count += 1
        #md = MD(frame.shape[0], frame.shape[1], frame.shape[2], byte_size, self.count) # create meta data (string type) to store image description
        self.sem.acquire()
        #memmove(md_buf, addressof(md), sizeof(md))    # same image meta data struct in shared memory
        #self.md_buf[:] = md                            # was #bytes(md_buf)
        if len(frame.shape) == 2:
            self.md_buf[:] = MD(frame.shape[0], frame.shape[1], 1, byte_size, self.count)
            #print("------------------------------------------- 2D ------------------------------------")
        else:
            self.md_buf[:] = MD(frame.shape[0], frame.shape[1], frame.shape[2], byte_size, self.count)
        self.shm_buf[:] = frame.tobytes()
        self.sem.release()

    def release(self):
        self.sem.acquire()

        self.md_buf.close()
        unlink_shared_memory(self.shm_name + '-meta')

        self.shm_buf.close()
        unlink_shared_memory(self.shm_name)

        self.sem.release()
        self.sem.close()

    def lock(self):
        self.sem.acquire()

    def unlock(self):
        self.sem.release()
    '''
    # Usage: 
    shm_w = ShmWrite('abc')    # create shared memory "abc" for write
    # get frame from somewhere
    shm_w.add(frame)           # put frame data in shared memory
    shm_w.release()
    '''

class ShmRead:
    def __init__(self, name):
        self.shm_buf = None
        self.md_buf = None
        self.tmp_md = create_string_buffer(sizeof(MD)) # template
        self.last = -1

        while not self.md_buf:
            try:
                print("Waiting for MetaData shared memory to be available.")
                md_region = SharedMemory(name + '-meta')
                self.md_buf = mmap.mmap(md_region.fd, sizeof(MD))
                md_region.close_fd()
                sleep(1)
            except ExistentialError:
                sleep(1)

        self.shm_name = name
        self.sem = Semaphore(name, 0)

    def get(self, fg=False):
        md = MD()

        self.sem.acquire()
        self.tmp_md[:] = self.md_buf
        memmove(addressof(md), self.tmp_md, sizeof(md))
        self.sem.release()

        #if fg and md.count == self.last: return None
        self.last = md.count

        while not self.shm_buf:
            try:
                print("Waiting for Data shared memory to be available.")
                shm_region = SharedMemory(name=self.shm_name)
                self.shm_buf = mmap.mmap(shm_region.fd, md.size)
                shm_region.close_fd()
                sleep(1)
            except ExistentialError:
                sleep(1)

        self.sem.acquire()
        f = np.ndarray(shape=(md.shape_0, md.shape_1, md.shape_2), dtype='uint8', buffer=self.shm_buf)
        self.sem.release()
        return f

    def release(self):
        self.md_buf.close()
        self.shm_buf.close()

    '''
    # Usage:
    shm_r = ShmRead('abc')
    f = shm_r.get()
    cv2.imshow('frame', f)
    ....
    shm_r.release()
    '''
