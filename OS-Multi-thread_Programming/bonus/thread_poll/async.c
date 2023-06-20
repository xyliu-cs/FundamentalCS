
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

task_queue *my_queue;
pthread_mutex_t queue_lock;
pthread_cond_t wakeup_call;

void *starter(void *_args)
{
	// never finish execution
	while (1) {
		Task *this_task;
		pthread_mutex_lock(&queue_lock);
		while (my_queue->size == 0) {
			pthread_cond_wait(&wakeup_call, &queue_lock);
		}

		this_task = my_queue->head;
		DL_DELETE(my_queue->head, my_queue->head);
		my_queue->size -= 1;

		pthread_mutex_unlock(&queue_lock);

		// execute
		this_task->target_function(this_task->arg);
	}
}

/** TODO: create num_threads threads and initialize the thread pool **/
void async_init(int num_threads)
{
	pthread_t pool[num_threads];
	int i, res;
	pthread_mutex_init(&queue_lock, NULL);
	pthread_cond_init(&wakeup_call, NULL);

	my_queue = (task_queue *)malloc(sizeof(task_queue));
	assert(my_queue);
	my_queue->head = NULL;
	my_queue->size = 0;

	for (i = 0; i < num_threads; i++) {
		res = pthread_create(&pool[i], NULL, &starter, NULL);
		if (res != 0) {
			perror("Failure to create the thread");
		}
	}
	return;
}

/** TODO: rewrite it to support thread pool **/
void async_run(void (*handler)(int), int args)
{
	Task *ins_task = (Task *)malloc(sizeof(Task));
	assert(ins_task);
	ins_task->target_function = handler;
	ins_task->arg = args;
	ins_task->prev = NULL;
	ins_task->next = NULL;

	pthread_mutex_lock(&queue_lock);

	DL_APPEND(my_queue->head, ins_task);

	my_queue->size += 1;

	pthread_mutex_unlock(&queue_lock);
	pthread_cond_signal(&wakeup_call);

	return;

	// handler(args);
}