#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)

/* If WIFSIGNALED(STATUS), the terminating signal.  */
#define __WTERMSIG(status) ((status)&0x7f)

/* If WIFSTOPPED(STATUS), the signal that stopped the child.  */
#define __WSTOPSIG(status) __WEXITSTATUS(status)

/* Nonzero if STATUS indicates normal termination.  */
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)

/* Nonzero if STATUS indicates termination by a signal.  */
#define __WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)

/* Nonzero if STATUS indicates the child is stopped.  */
#define __WIFSTOPPED(status) (((status)&0xff) == 0x7f)

struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int wo_stat; // Note that here wo_stat is int rather than *int!
	struct rusage *wo_rusage;
	wait_queue_entry_t child_wait;
	int notask_error;
};

static struct task_struct *pcs_info;

// use exported functions
extern long do_wait(struct wait_opts *wo);
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);

extern pid_t kernel_clone(struct kernel_clone_args *kargs);
extern struct filename *getname_kernel(const char *filename);

int signal_display(int e_status)
{
	// normal termination
	if (__WIFEXITED(e_status)) {
		printk("[program2] : Child process gets normal termination");
		printk("[program2] : The return signal is 0\n");
		return 0;
	}
	// stop
	else if (__WIFSTOPPED(e_status)) {
		int stop_sig = __WSTOPSIG(e_status);
		if (stop_sig == 19) {
			printk("[program2] : Get SIGSTOP signal");
		} else {
			printk("[program2] : Get unrecognizable signal for STOP");
		}
		return stop_sig;
	} else if (__WIFSIGNALED(e_status)) {
		int r_status = __WTERMSIG(e_status);
		// hang up
		if (r_status == 1) {
			printk("[program2] : Get SIGHUP signal");
		}
		// interrupt
		else if (r_status == 2) {
			printk("[program2] : Get SIGINT signal");
		}
		// quit
		else if (r_status == 3) {
			printk("[program2] : Get SIGQUIT signal");
		}
		// illegal instruction
		else if (r_status == 4) {
			printk("[program2] : Get SIGILL signal");
		}
		// trap
		else if (r_status == 5) {
			printk("[program2] : Get SIGTRAP signal");
		}
		// abort
		else if (r_status == 6) {
			printk("[program2] : Get SIGABRT signal");
		}
		// bus
		else if (r_status == 7) {
			printk("[program2] : Get SIGBUS signal");
		}
		// floating point error
		else if (r_status == 8) {
			printk("[program2] : Get SIGFPE signal");
		}
		// kill
		else if (r_status == 9) {
			printk("[program2] : Get SIGKILL signal");
		}
		// segment fault
		else if (r_status == 11) {
			printk("[program2] : Get SIGSEGV signal");
		}
		// pipe
		else if (r_status == 13) {
			printk("[program2] : Get SIGPIPE signal");
		}
		// alarm
		else if (r_status == 14) {
			printk("[program2] : Get SIGALRM signal");
		}
		// terminate
		else if (r_status == 15) {
			printk("[program2] : Get SIGTERM signal");
		} else {
			printk("[program2] : Get unrecognizable signal");
		}
		return r_status;
	} else {
		printk("[program2] : Child process continued\n");
		return 0;
	}
}

// implement wait function, return child exit status
int my_wait(pid_t pid)
{
	int status = 0;
	long do_wait_ret;
	int ret;

	struct wait_opts wo = { .wo_type = PIDTYPE_PID,
				.wo_pid = find_get_pid(pid),
				.wo_info = NULL,
				.wo_flags = WEXITED | WUNTRACED,
				.wo_stat = status,
				.wo_rusage = NULL };

	do_wait_ret = do_wait(&wo);
	ret = wo.wo_stat;
	// printk("[program2] : mywait: do_wait return signal is %ld\n", do_wait_ret);
	// printk("[program2] : mywait: the return signal is %d\n", ret);

	put_pid(wo.wo_pid);
	return ret;
}

int my_exec(void)
{
	int ret;
	const char *path = "/tmp/test";
	struct filename *openf;

	printk("[program2] : Child process");

	openf = getname_kernel(path);
	ret = do_execve(openf, NULL, NULL);

	// do_execve returns NULL upon success
	if (!ret) {
		return 0;
	}

	// Error handling
	else {
		do_exit(ret);
	}
}

// implement fork function
int my_fork(void *argc)
{
	int exit_status;
	int ret_value;
	pid_t chld_pid;

	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	struct kernel_clone_args kargs = {};

	// set default sigaction for current process
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	// set kernel_clone_args parameters
	// let the child process execute my_exec function
	kargs.flags = SIGCHLD;
	kargs.stack = (unsigned long)&my_exec;
	kargs.stack_size = 0;
	kargs.parent_tid = NULL;
	kargs.child_tid = NULL;
	kargs.tls = 0;
	kargs.exit_signal = SIGCHLD;

	/* fork a process using kernel_clone or kernel_thread */
	/* execute a test program in child process */

	chld_pid = kernel_clone(&kargs);
	printk("[program2] : The child process has pid = %d\n", chld_pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       (int)current->pid);

	/* wait until child process terminates */
	ret_value = my_wait(chld_pid);
	exit_status = signal_display(ret_value);
	if (exit_status != 0) {
		printk("[program2] : Child process terminated");
		printk("[program2] : The return signal is %d\n", exit_status);
	}

	return 0;
}

static int __init program2_init(void)
{
	/* write your code here */

	printk("[program2] : Module_init {LiuXiaoyuan} {120040051}\n");
	/* create a kernel thread to run my_fork */

	printk("[program2] : Module_init create kthread start\n");
	pcs_info = kthread_create(&my_fork, NULL, "My Thread");

	if (!IS_ERR(pcs_info)) {
		printk("[program2] : Module_init kthread start\n");
		wake_up_process(pcs_info);
	} else {
		PTR_ERR(pcs_info);
	}
	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
