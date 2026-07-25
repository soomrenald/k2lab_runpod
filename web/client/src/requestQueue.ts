type PendingRequest<T> = {
  operation: () => Promise<T>;
  priority: number;
  sequence: number;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
};

export class SerialRequestLane {
  private pending: PendingRequest<unknown>[] = [];
  private running = false;
  private sequence = 0;

  run<T>(operation: () => Promise<T>, priority = 10): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.push({
        operation,
        priority,
        sequence: this.sequence,
        resolve,
        reject,
      } as PendingRequest<unknown>);
      this.sequence += 1;
      this.pending.sort((left, right) => (
        left.priority - right.priority || left.sequence - right.sequence
      ));
      void this.drain();
    });
  }

  private async drain() {
    if (this.running) return;
    this.running = true;
    try {
      while (this.pending.length) {
        const request = this.pending.shift();
        if (!request) continue;
        try {
          request.resolve(await request.operation());
        } catch (caught) {
          request.reject(caught);
        }
      }
    } finally {
      this.running = false;
    }
  }
}

export const controlRequestLane = new SerialRequestLane();
export const bulkRequestLane = new SerialRequestLane();
