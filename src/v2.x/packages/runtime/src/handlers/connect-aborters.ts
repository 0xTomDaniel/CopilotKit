type ConnectAborter = () => void;

type GlobalWithConnectAborters = typeof globalThis & {
  __copilotkitConnectAbortersByThread?: Map<string, Set<ConnectAborter>>;
};

const globalWithAborters = globalThis as GlobalWithConnectAborters;

export const CONNECT_ABORTERS_BY_THREAD =
  globalWithAborters.__copilotkitConnectAbortersByThread ??
  (globalWithAborters.__copilotkitConnectAbortersByThread = new Map());
