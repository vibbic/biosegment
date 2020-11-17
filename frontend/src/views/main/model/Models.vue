<template>
  <div>
    <v-toolbar light>
      <v-toolbar-title>
        Manage Models
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn color="primary" to="/main/models/create">Create Model</v-btn>
    </v-toolbar>
    <v-data-table :headers="headers" :items="models" item-key="name">
      <template v-slot:item="{ item }">
        <tr>
          <td>{{ item.title }}</td>
          <td>
            <v-btn text :to="{name: 'main-models-edit', params: {id: item.id}}">
              <v-icon>edit</v-icon>
            </v-btn>
            <v-btn text @click="deleteModel(item.id)">
              <v-icon>delete</v-icon>
            </v-btn>
          </td>
        </tr>
      </template>
    </v-data-table>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Store } from 'vuex';
import { Model } from '@/api';
import { readModels } from '@/store/model/getters';
import { dispatchGetModels, dispatchDeleteModel } from '@/store/model/actions';

@Component
export default class ModelModels extends Vue {
  public headers = [
    {
      text: 'Title',
      sortable: true,
      value: 'title',
      align: 'left',
    },
    {
      text: 'Actions',
      value: 'id',
    },
  ];
  get models() {
    return readModels(this.$store);
  }

  public async mounted() {
    await dispatchGetModels(this.$store);
  }

  public async deleteModel(id: number) {
    await dispatchDeleteModel(this.$store, {id});
  }
}
</script>
