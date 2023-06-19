#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

const char *POSIX_SIG[] = { NULL,     "SIGHUP",	 "SIGINT",  "SIGQUIT",
			    "SIGILL", "SIGTRAP", "SIGABRT", "SIGBUS",
			    "SIGFPE", "SIGKILL", NULL,	    "SIGSEGV",
			    NULL,     "SIGPIPE", "SIGALRM", "SIGTERM" };

// const char *SIG_NAME[] = { NULL,
// 			   "hangup",
// 			   "interrupt",
// 			   "quit",
// 			   "illegal instruction",
// 			   "trap",
// 			   "abort",
// 			   "bus error",
// 			   "floating-point exception",
// 			   "kill",
// 			   NULL,
// 			   "segment fault",
// 			   NULL,
// 			   "pipe",
// 			   "alarm",
// 			   "terminate" };

int main(int argc, char *argv[])
{
	char buf[50] = "Original test strings";
	pid_t pid;
	pid_t rt_pid;
	int chld_status;
	int signum;
	printf("Process start to fork\n");

	/* fork a child process */
	pid = fork();
	// fork failure
	if (pid == -1) {
		perror("fork failed");
		exit(1);
	} else {
		/* execute test program */
		// fork() returns 0: Executing Child Process
		if (pid == 0) {
			int i;
			char *arg[argc];

			printf("I'm the Child Process, my pid = %d\n",
			       getpid());
			// copy the parameters for the main function
			// skip the first file name which is this file
			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;

			printf("Child process start to execute test program:\n");
			execve(arg[0], arg, NULL);

			// Exit Anomaly
			printf("Continue to execute the Child process!!\n");
			perror("exit failure");
			exit(EXIT_FAILURE);
		}

		// Executing Parent Process: fork() returns the pid of the Child Process
		else {
			printf("I'm the Parent Process, my pid = %d\n",
			       getpid());
			/* wait for child process terminates */
			// waitpid returns the child pid
			rt_pid = waitpid(pid, &chld_status, WUNTRACED);

			/* check child process's termination status */
			if (rt_pid > 0) {
				printf("Parent process receives SIGCHLD signal\n");
				if (WIFEXITED(chld_status)) {
					printf("Normal termination with EXIT STATUS = %d\n",
					       WEXITSTATUS(chld_status));
				} else if (WIFSIGNALED(chld_status)) {
					int sig = WTERMSIG(chld_status);
					printf("Child process get %s signal\n",
					       POSIX_SIG[sig]);
				} else if (WIFSTOPPED(chld_status)) {
					int s_sig = WSTOPSIG(chld_status);
					printf("Child process get SIGSTOP signal\n");
				} else {
					printf("------------ERROR! EXITING PROGRAM------------\n");
				}
				exit(0);
			} else {
				printf("Error\n");
				exit(1);
			}
		}
	}
	return 0;
}
